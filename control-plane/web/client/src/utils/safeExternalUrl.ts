/** Returns the normalized validated URL for a safe external link, else null.
 * Mirrors server validation and re-checks stored rows that may predate it. */
export function safeExternalUrl(
    raw: string | null | undefined,
    allowMailto = false,
): string | null {
    if (!raw) return null;
    try {
        const parsed = new URL(raw);
        if (
            (parsed.protocol !== "http:" && parsed.protocol !== "https:" &&
                !(allowMailto && parsed.protocol === "mailto:")) ||
            (parsed.protocol !== "mailto:" && !parsed.hostname) ||
            parsed.username ||
            parsed.password
        ) {
            return null;
        }
        return parsed.href;
    } catch {
        return null;
    }
}
