import React from 'react';
import type { ReasonerDefinition } from '../types/haxen';
import { Badge } from '@/components/ui/badge';
import { WatsonxAi } from '@/components/ui/icon-bridge';

interface ReasonersListProps {
  reasoners: ReasonerDefinition[];
}

const ReasonersList: React.FC<ReasonersListProps> = ({ reasoners }) => {
  if (!reasoners || reasoners.length === 0) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <WatsonxAi className="h-4 w-4 text-muted-foreground" />
          <h4 className="text-sm font-medium">Reasoners (0)</h4>
        </div>
        <p className="text-body-small">No reasoners available.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <WatsonxAi className="h-4 w-4 text-muted-foreground" />
        <h4 className="text-sm font-medium">Reasoners ({reasoners.length})</h4>
      </div>
      <div className="flex flex-wrap gap-2">
        {reasoners.map((reasoner) => (
          <Badge key={reasoner.id} variant="secondary" className="text-xs">
            {reasoner.id}
          </Badge>
        ))}
      </div>
    </div>
  );
};

export default ReasonersList;
