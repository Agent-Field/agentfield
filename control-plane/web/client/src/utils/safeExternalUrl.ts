/** Returns a URL when it is safe to render as an external link, else null.
 * Mirrors server validation and re-checks stored rows that may predate it. */
export function safeExternalUrl(raw: string | null | undefined): string | null {
    if (!raw) return null;
    try {
        const parsed = new URL(raw);
        if (
            (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
            !parsed.hostname ||
            parsed.username ||
            parsed.password
        ) {
            return null;
        }
        return raw;
    } catch {
        return null;
    }
}
