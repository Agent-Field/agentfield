export const CODEAF_VERSION = 'v0.8.2';
export const CODEAF_BASE_URL = `https://github.com/Agent-Field/CodeAF/releases/download/${CODEAF_VERSION}`;
export const CODEAF_MAX_DOWNLOAD_BYTES = 128 * 1024 * 1024;

// Hashes cover the raw executables, not an archive or a downloaded manifest.
export const CODEAF_SHA256: Readonly<Record<string, string>> = Object.freeze({
  'codeaf-darwin-amd64': 'a476f1ce44b21f8016583a65c49b21497d10713bcd72c33478b3bd12e52446d7',
  'codeaf-darwin-arm64': 'e61adac060bf2345c08a3e23ae83239871c3050eadac046879f5f3176270c4f1',
  'codeaf-linux-amd64': '681d9813f2517addc86fb644d313e3f7cf0ab60be708c8e8b13d77fb16a6798c',
  'codeaf-linux-arm64': 'c6375d58d198d9e1a84880d8b4324ee6c067a0e5e351af115ca5506b1ffa1c78',
  'codeaf-windows-amd64.exe': 'ff80f6f0c89c209fc809e960c70a1eb3518ccfc06775d587d5a187edd5d5222d',
  'codeaf-windows-arm64.exe': 'bfc0b386e5248e7e6756f097b372f72e4e955ee1e19e9216ccafee47064b2b80',
});
