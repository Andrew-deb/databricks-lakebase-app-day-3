-- Trace table for Agent Bricks MCP HTTP requests.
-- Run this SQL against Lakebase before deploying the MCP server.

CREATE TABLE IF NOT EXISTS agent_mcp_traces (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    duration_ms DECIMAL(12, 2) NOT NULL,
    method VARCHAR(16) NOT NULL,
    path VARCHAR(512) NOT NULL,
    status_code INTEGER NOT NULL,
    user_email VARCHAR(255),
    mcp_session_id VARCHAR(255),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_mcp_traces_session_id ON agent_mcp_traces (session_id);

CREATE INDEX IF NOT EXISTS idx_agent_mcp_traces_started_at ON agent_mcp_traces (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_mcp_traces_user_email ON agent_mcp_traces (user_email);