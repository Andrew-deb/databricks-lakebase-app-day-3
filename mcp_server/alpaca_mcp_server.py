"""
Alpaca Markets paper-trading MCP server.

Exposes paper-trading tools over MCP (Model Context Protocol) so a
Databricks Agent Bricks agent can call them like any other tool:
    - get_quote(symbol)
    - stage_trade(account_id, symbol, side, quantity)
    - execute_trade(account_id, symbol, side, quantity, confirmation_code)
    - get_positions(account_id)
    - get_account_summary(account_id)
    - get_order_history(account_id, limit)
    - get_balance(account_id)
    - get_current_user()

These tools are backed by Alpaca Markets' real, hosted paper-trading
account (see alpaca_broker.py), so students can safely wire an Agent
Bricks agent to place real (but fake-money) trades without a real
brokerage account or risk of real money moving. account_id is accepted
for signature compatibility but is not used to select an account - Alpaca
paper trading is one account per API key pair.

Swap-in-a-real-broker note: to point this at a different broker instead,
keep the same 5 tool signatures below and replace the alpaca_broker.*
calls inside each tool with calls to that broker's SDK/API - the MCP
surface for the agent does not need to change. The original Lakebase-
simulated engine is preserved in paper_broker.py for reference.

Deploy this as its own Databricks App (same app.yaml + FastMCP entrypoint
pattern documented at
https://docs.databricks.com/aws/en/agents/mcp-tools/custom-mcp), separate
from the dashboard app, so an Agent Bricks agent (or any MCP client) can
register its URL as an external MCP server.

Run locally:
    python alpaca_mcp_server.py
"""

import os
import logging
import secrets
import threading
import time
import uuid
import asyncio
from contextvars import ContextVar
from datetime import datetime, timezone
import json

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from sentence_transformers import SentenceTransformer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

import alpaca_broker
import massive_broker
import lakebase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpaca-mcp-server")

# Load embedding model once at startup
_embedding_model = None

def get_embedding_model():
    """Lazy-load the embedding model (expensive operation, only on first use)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

# Table names from environment variables
NEWS_TABLE_NAME = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("EMBEDDINGS_TABLE_NAME", "ticker_news_embeddings")
CHUNK_EMBEDDINGS_TABLE_NAME = os.environ.get("CHUNK_EMBEDDINGS_TABLE_NAME", "ticker_news_chunk_embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TRACE_TABLE_NAME = os.environ.get("TRACE_TABLE_NAME", "agent_mcp_traces")

# Context variable to store request headers for accessing end-user identity
_request_context: ContextVar[dict] = ContextVar('request_context', default={})

_STAGED_TRADE_TTL_SECONDS = 10 * 60
_staged_trades: dict[str, dict] = {}
_staged_trades_lock = threading.Lock()
_session_results: dict[str, list[dict]] = {}
_session_results_lock = threading.Lock()


def _get_end_user_email() -> str:
    """Get the actual end user's email from request headers, or fallback to service principal."""
    # Try to get from X-Forwarded-User header (Databricks App context)
    headers = _request_context.get()
    forwarded_user = headers.get('x-forwarded-user')
    if forwarded_user:
        return forwarded_user
    
    # Fallback: use service principal (local development or non-App contexts)
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    return w.current_user.me().user_name or 'omakuandy@gmail.com'


mcp = FastMCP("alpaca-paper-trading")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Capture end-user identity for tools that need request headers."""
    async def dispatch(self, request: Request, call_next):
        session_id = (
            request.headers.get("mcp-session-id")
            or request.headers.get("x-agent-session-id")
            or str(uuid.uuid4())
        )
        headers = {
            'x-forwarded-user': request.headers.get('x-forwarded-user'),
            'x-forwarded-email': request.headers.get('x-forwarded-email'),
            'x-agent-session-id': session_id,
        }
        _request_context.set(headers)
        response = await call_next(request)
        response.headers.setdefault("x-agent-session-id", session_id)
        return response


def _json_default(value):
    """Convert SDK/Pydantic values into JSON-safe values for JSONB storage."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _result_payload(result) -> dict:
    """Convert a FastMCP result into a JSON-safe payload."""
    return {
        "is_error": getattr(result, "is_error", False),
        "structured_content": getattr(result, "structured_content", None),
        "content": getattr(result, "content", None),
    }


def _execution_outcome(tool_name: str, result, error_message: str | None) -> dict:
    """Build the standardized outcome recorded for every tool execution."""
    result_is_error = bool(getattr(result, "is_error", False))
    status = "error" if error_message or result_is_error else "success"
    if error_message:
        message = f"Tool {tool_name} errored: {error_message}"
    elif result_is_error:
        message = f"Tool {tool_name} returned an error result."
    else:
        message = f"Tool {tool_name} executed successfully."
    return {
        "tool_name": tool_name,
        "status": status,
        "message": message,
        "result": _result_payload(result) if result is not None else None,
        "error": error_message,
        "completed_at": datetime.now(timezone.utc),
    }


def _session_result_json(session_id: str, tool_name: str, result, error_message: str | None) -> str:
    """Append this outcome and return the bounded session result aggregate."""
    outcome = _execution_outcome(tool_name, result, error_message)
    with _session_results_lock:
        outcomes = _session_results.setdefault(session_id, [])
        outcomes.append(outcome)
        del outcomes[:-100]
        return json.dumps({"session_id": session_id, "tool_calls": outcomes}, default=_json_default)


def _trace_user_email() -> str | None:
    headers = _request_context.get()
    if headers.get("x-forwarded-user") or headers.get("x-forwarded-email"):
        return headers.get("x-forwarded-user") or headers.get("x-forwarded-email")
    try:
        from fastmcp.server.dependencies import get_http_headers
        headers = get_http_headers() or {}
        return headers.get("x-forwarded-user") or headers.get("x-forwarded-email")
    except Exception:
        return None


class TraceMiddleware(Middleware):
    """Persist tool names and the accumulated result for each MCP session."""
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        fastmcp_context = context.fastmcp_context
        session_id = (
            getattr(fastmcp_context, "session_id", None)
            or _request_context.get().get("x-agent-session-id")
            or str(uuid.uuid4())
        )
        request_id = getattr(fastmcp_context, "request_id", None) or str(uuid.uuid4())
        tool_name = context.message.name
        error_message = None
        result = None
        try:
            result = await call_next(context)
            return result
        except Exception as error:
            error_message = str(error)
            raise
        finally:
            session_result = _session_result_json(session_id, tool_name, result, error_message)
            await asyncio.to_thread(_write_trace, {
                "request_id": str(request_id),
                "session_id": str(session_id),
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "method": context.method,
                "path": "/mcp",
                "status_code": 500 if error_message or getattr(result, "is_error", False) else 200,
                "mcp_session_id": str(session_id),
                "tool_name": tool_name,
                "session_result": session_result,
                "user_email": _trace_user_email(),
                "error_message": error_message,
            })


def _write_trace(trace: dict) -> None:
    """Persist telemetry without allowing Lakebase failures to affect MCP."""
    try:
        lakebase.run_write(
            f"""
            INSERT INTO {TRACE_TABLE_NAME} (
                request_id, session_id, started_at, finished_at, duration_ms,
                method, path, status_code, user_email, mcp_session_id,
                tool_name, session_result, error_message
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s::jsonb, %s
            )
            """,
            (
                trace["request_id"],
                trace["session_id"],
                trace["started_at"],
                trace["finished_at"],
                trace["duration_ms"],
                trace["method"],
                trace["path"],
                trace["status_code"],
                trace["user_email"],
                trace["mcp_session_id"],
                trace["tool_name"],
                trace["session_result"],
                trace["error_message"],
            ),
        )
    except Exception:
        logger.exception("Failed to persist MCP trace in table %s", TRACE_TABLE_NAME)


mcp.add_middleware(TraceMiddleware())


@mcp.tool
def get_quote(symbol: str) -> dict:
    """
    Get the latest real quote for a stock ticker symbol from Massive.com.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        A dict with symbol, price, as_of (ISO timestamp), volume, change, and change_percent.
    """
    return massive_broker.get_quote(symbol)


@mcp.tool
def stage_trade(account_id: str, symbol: str, side: str, quantity: float) -> dict:
    """
    Stage a paper trade without placing an order. Looks up the current quote,
    estimates the trade cost, summarizes the trade, and returns a five-digit
    confirmation code for execute_trade.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account (Alpaca paper trading is one account per
            API key pair).
        symbol: Stock ticker symbol, e.g. "AAPL".
        side: "BUY" or "SELL".
        quantity: Number of shares to trade (must be positive).

    Returns:
        A dict with the staged trade details, estimated cost, summary, and
        confirmation_code. The code expires after 10 minutes and is single-use.
    """
    symbol = symbol.strip().upper()
    side = side.strip().upper()
    quantity = float(quantity)

    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity!r}")

    quote = massive_broker.get_quote(symbol)
    estimated_cost = round(quote["price"] * quantity, 2)
    staged_trade = {
        "account_id": account_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": quote["price"],
        "estimated_cost": estimated_cost,
        "created_at": time.time(),
    }

    with _staged_trades_lock:
        now = time.time()
        expired_codes = [
            code for code, trade in _staged_trades.items()
            if now - trade["created_at"] > _STAGED_TRADE_TTL_SECONDS
        ]
        for code in expired_codes:
            del _staged_trades[code]
        confirmation_code = f"{secrets.randbelow(100000):05d}"
        while confirmation_code in _staged_trades:
            confirmation_code = f"{secrets.randbelow(100000):05d}"
        _staged_trades[confirmation_code] = staged_trade

    return {
        "status": "staged",
        "account_id": account_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "quoted_price": quote["price"],
        "estimated_cost": estimated_cost,
        "quote_as_of": quote.get("as_of"),
        "summary": f"{side} {quantity:g} {symbol} at approximately ${quote['price']:.2f} per share, estimated cost ${estimated_cost:.2f}.",
        "confirmation_code": confirmation_code,
        "expires_in_seconds": _STAGED_TRADE_TTL_SECONDS,
    }


@mcp.tool
def execute_trade(
    account_id: str,
    symbol: str,
    side: str,
    quantity: float,
    confirmation_code: str,
) -> dict:
    """Execute a previously staged paper trade after code confirmation.

    The confirmation code must be the five-digit code returned by stage_trade
    for the same account, symbol, side, and quantity.
    """
    symbol = symbol.strip().upper()
    side = side.strip().upper()
    quantity = float(quantity)
    confirmation_code = confirmation_code.strip()

    if not (len(confirmation_code) == 5 and confirmation_code.isdigit()):
        return {
            "status": "confirmation_required",
            "message": "Supply the five-digit confirmation_code returned by stage_trade.",
        }

    with _staged_trades_lock:
        staged_trade = _staged_trades.get(confirmation_code)
        if staged_trade is None:
            return {
                "status": "invalid_confirmation",
                "message": "The confirmation code is invalid, expired, or already used.",
            }
        if time.time() - staged_trade["created_at"] > _STAGED_TRADE_TTL_SECONDS:
            del _staged_trades[confirmation_code]
            return {
                "status": "invalid_confirmation",
                "message": "The confirmation code has expired. Stage the trade again.",
            }
        if any(
            staged_trade[key] != value
            for key, value in {
                "account_id": account_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            }.items()
        ):
            return {
                "status": "trade_mismatch",
                "message": "The confirmation code does not match the staged trade details.",
            }

        # Consume before the external call so retries cannot submit twice.
        del _staged_trades[confirmation_code]

    order = alpaca_broker.place_order(account_id, symbol, side, quantity)

    # Refresh the user's watchlist only after Alpaca accepts the transaction.
    try:
        watchlist_update = add_to_watchlist(symbol, _get_end_user_email())
    except Exception as error:
        logger.exception("Trade succeeded but watchlist update failed for %s", symbol)
        watchlist_update = {
            "status": "error",
            "message": f"Trade succeeded, but watchlist update failed: {error}",
        }

    order["watchlist_update"] = watchlist_update
    return order


@mcp.tool
def get_positions(account_id: str) -> list[dict]:
    """
    Get all open positions for the Alpaca paper trading account.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.

    Returns:
        A list of dicts, each with symbol, quantity, avg_cost, updated_at.
    """
    return alpaca_broker.get_positions(account_id)


@mcp.tool
def get_account_summary(account_id: str) -> dict:
    """
    Get a full account summary for the Alpaca paper trading account: cash
    balance, open positions marked-to-market, total market value, and
    total equity (cash + market value).

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.

    Returns:
        A dict with account_id, cash_balance, positions, market_value,
        total_equity.
    """
    return alpaca_broker.get_account_summary(account_id)


@mcp.tool
def get_order_history(account_id: str, limit: int = 50) -> list[dict]:
    """
    Get recent orders for the Alpaca paper trading account, most recent first.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.
        limit: Max number of orders to return (default 50).

    Returns:
        A list of dicts, each with id, symbol, side, quantity, price,
        notional, status, created_at.
    """
    return alpaca_broker.get_order_history(account_id, limit)


@mcp.tool
def get_balance(account_id: str) -> dict:
    """
    Get the current cash balance and buying power for the Alpaca paper 
    trading account.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.

    Returns:
        A dict with account_id, cash_balance, buying_power, and currency.
    """
    return alpaca_broker.get_account_summary(account_id)


@mcp.tool
def get_current_user() -> dict:
    """
    Get information about the currently authenticated end user accessing the MCP server.
    
    When running as a Databricks App, this returns the actual end user making the
    request (from X-Forwarded-User header), not the service principal running the app.

    Returns:
        A dict with user_name (email from X-Forwarded-User header), 
        forwarded_email, and source ("request_header" or "service_principal").
    """
    try:
        # First, try to get the end user from the request headers
        # Databricks injects X-Forwarded-User with the actual user's email
        headers = _request_context.get()
        forwarded_user = headers.get('x-forwarded-user')
        forwarded_email = headers.get('x-forwarded-email')
        
        if forwarded_user:
            return {
                "status": "success",
                "user_name": forwarded_user,
                "forwarded_email": forwarded_email,
                "source": "request_header",
            }
        
        # Fallback: return the service principal if headers aren't available
        # (e.g., when running locally or in non-App contexts)
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        user = w.current_user.me()
        return {
            "status": "success",
            "user_name": user.user_name,
            "display_name": user.display_name,
            "active": user.active,
            "source": "service_principal",
        }
    except Exception as e:
        logger.exception("Failed to get current user")
        return {
            "status": "error",
            "message": f"Failed to get current user: {str(e)}",
        }


@mcp.tool
def add_to_watchlist(symbol: str, email: str = 'omakuandy@gmail.com') -> dict:
    """
    Add a stock to the watchlist by fetching its current quote from Massive.com
    and storing it in the Lakebase watchlist table.
    
    Uses the authenticated user's email as the user_id.
    
    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
    
    Returns:
        A dict with the quote data and confirmation that it was added to the watchlist.
    """
    try:
        # Get the actual end user's email (not the service principal)
        user_email = email
        
        # Get quote from Massive.com
        quote = massive_broker.get_quote(symbol)
        
        # Store in Lakebase watchlist table
        sql = """
        INSERT INTO watchlist (email, symbol, latest_price, updated_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (email, symbol) 
        DO UPDATE SET 
            latest_price = EXCLUDED.latest_price,
            updated_at = NOW()
        """
        
        lakebase.run_write(
            sql,
            (
                user_email,
                quote["symbol"],
                quote["price"]
            ),
        )
        
        return {
            "status": "success",
            "message": f"Added {symbol} to watchlist for {user_email}",
            "user_email": user_email,
            "quote": quote,
        }
    except Exception as e:
        logger.exception(f"Failed to add {symbol} to watchlist")
        return {
            "status": "error",
            "message": f"Failed to add {symbol} to watchlist: {str(e)}",
        }


@mcp.tool
def get_watchlist(limit: int = 100, email: str = 'omakuandy@gmail.com') -> dict:
    """
    Retrieve all stocks in the authenticated user's watchlist from Lakebase.
    
    Uses the authenticated user's email as the user_id.
    
    Args:
        limit: Maximum number of entries to return (default: 100).
        email: authenticate user's email
    
    Returns:
        A dict with watchlist entries sorted by most recently added.
    """
    try:
        # Get the actual end user's email (not the service principal)
        
        sql = """
        SELECT 
            symbol,
            latest_price,
            updated_at
        FROM watchlist
        WHERE email = %s
        LIMIT %s
        """
        
        rows = lakebase.run_query(sql, (email, limit))
        
        return {
            "status": "success",
            "user_email": email,
            "count": len(rows),
            "watchlist": rows,
        }
    except Exception as e:
        logger.exception(f"Failed to retrieve watchlist")
        return {
            "status": "error",
            "message": f"Failed to retrieve watchlist: {str(e)}",
        }


@mcp.tool
def remove_from_watchlist(symbol: str) -> dict:
    """
    Remove a stock from the authenticated user's watchlist.
    
    Uses the authenticated user's email as the user_id.
    
    Args:
        symbol: Stock ticker symbol to remove, e.g. "AAPL".
    
    Returns:
        A dict with status and confirmation message.
    """
    try:
        # Get the actual end user's email (not the service principal)
        user_email = _get_end_user_email()
        
        symbol = symbol.strip().upper()
        
        sql = """
        DELETE FROM watchlist
        WHERE email = %s AND symbol = %s
        """
        
        rows_affected = lakebase.run_write(sql, (user_email, symbol))
        
        if rows_affected > 0:
            return {
                "status": "success",
                "message": f"Removed {symbol} from watchlist",
                "symbol": symbol,
                "user_email": user_email,
            }
        else:
            return {
                "status": "not_found",
                "message": f"{symbol} was not in the watchlist",
                "symbol": symbol,
                "user_email": user_email,
            }
    except Exception as e:
        logger.exception(f"Failed to remove {symbol} from watchlist")
        return {
            "status": "error",
            "message": f"Failed to remove {symbol} from watchlist: {str(e)}",
        }


@mcp.tool
def vector_search(query: str, limit: int = 10, search_chunks: bool = True) -> dict:
    """
    Semantic search over ticker news using vector embeddings.
    
    Accepts a text query, computes its embedding, and returns the most similar
    documents and chunks from Lakebase using pgvector's cosine similarity.
    
    Args:
        query: Natural language search query (e.g. "tech company earnings")
        limit: Maximum number of results to return (default 10)
        search_chunks: Whether to search chunk-level embeddings in addition to documents
    
    Returns:
        A dict with query, documents, chunks, and model name
    """
    if not query or not query.strip():
        return {"error": "Query text is required"}
    
    try:
        # Compute embedding for the query
        model = get_embedding_model()
        query_embedding = model.encode(query)
        
        # Convert to list for JSON serialization and postgres array format
        embedding_list = query_embedding.tolist()
        
        # Search document-level embeddings
        doc_results = lakebase.run_query(
            f"""
            SELECT 
                e.id,
                e.ticker,
                e.title,
                e.published_utc,
                e.model_name,
                1 - (e.embedding <=> %s::vector) as similarity,
                d.description,
                d.article_url,
                d.sentiment
            FROM {EMBEDDINGS_TABLE_NAME} e
            LEFT JOIN {NEWS_TABLE_NAME} d ON e.id = d.id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (str(embedding_list), str(embedding_list), limit),
        )
        
        chunk_results = []
        if search_chunks:
            # Search chunk-level embeddings
            chunk_results = lakebase.run_query(
                f"""
                SELECT 
                    c.id,
                    c.article_id,
                    c.ticker,
                    c.chunk_index,
                    c.chunk_text,
                    c.model_name,
                    1 - (c.embedding <=> %s::vector) as similarity,
                    d.title,
                    d.article_url,
                    d.published_utc
                FROM {CHUNK_EMBEDDINGS_TABLE_NAME} c
                LEFT JOIN {NEWS_TABLE_NAME} d ON c.article_id = d.id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (str(embedding_list), str(embedding_list), limit),
            )
        
        return {
            "query": query,
            "documents": doc_results,
            "chunks": chunk_results,
            "model": EMBEDDING_MODEL
        }
        
    except Exception as e:
        logger.exception("Vector search failed")
        return {"error": str(e)}


if __name__ == "__main__":
    # Add middleware to capture request headers for end-user identity
    # This must be done before mcp.run() is called
    if hasattr(mcp, 'app') and mcp.app is not None:
        mcp.app.add_middleware(RequestContextMiddleware)
    
    # Databricks Apps route external HTTP traffic to this port via app.yaml;
    # streamable-http is the transport Databricks' MCP client/gateway expects
    # (see the "Host your own MCP" doc linked in the module docstring above).
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)
