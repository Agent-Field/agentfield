-- Migration: Replay linkage on inbound_events
-- Description: Records the ID of the original event a replay was cloned from.
--              Without this, a replayed event is indistinguishable from a
--              fresh delivery in the API + UI surfaces — operators have no
--              way to navigate back from "this replay" to "the original
--              webhook delivery." idempotency_key is cleared on replays
--              (so providers' dedup index doesn't reject re-dispatch), so
--              we need an explicit pointer rather than re-deriving from key.
-- Created: 2026-04-29

-- +goose Up
-- +goose StatementBegin
ALTER TABLE inbound_events
    ADD COLUMN replay_of TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_inbound_events_replay_of
    ON inbound_events(replay_of)
    WHERE replay_of <> '';
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP INDEX IF EXISTS idx_inbound_events_replay_of;
ALTER TABLE inbound_events DROP COLUMN replay_of;
-- +goose StatementEnd
