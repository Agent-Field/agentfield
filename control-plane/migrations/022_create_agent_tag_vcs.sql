-- +goose Up
CREATE TABLE IF NOT EXISTS agent_tag_vcs (
    id              BIGSERIAL PRIMARY KEY,
    agent_id        TEXT NOT NULL UNIQUE,
    agent_did       TEXT NOT NULL,
    vc_id           TEXT NOT NULL UNIQUE,
    vc_document     TEXT NOT NULL,
    signature       TEXT,
    issued_at       TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at      TIMESTAMP WITH TIME ZONE,
    revoked_at      TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_tag_vcs_agent_did ON agent_tag_vcs(agent_did);
CREATE INDEX IF NOT EXISTS idx_agent_tag_vcs_vc_id ON agent_tag_vcs(vc_id);

-- +goose Down
DROP TABLE IF EXISTS agent_tag_vcs;
