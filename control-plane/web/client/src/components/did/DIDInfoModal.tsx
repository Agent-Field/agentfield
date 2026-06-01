interface DIDInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  nodeId: string;
}

export function DIDInfoModal({ isOpen, onClose, nodeId: _nodeId }: DIDInfoModalProps) {
  if (!isOpen) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div className="bg-background rounded-lg p-6 shadow-lg" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-lg font-semibold mb-2">DID Information</h2>
        <button onClick={onClose} className="mt-4 text-sm text-muted-foreground hover:text-foreground">
          Close
        </button>
      </div>
    </div>
  );
}
