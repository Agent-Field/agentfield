import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Function, Information } from "@/components/ui/icon-bridge";
import React, { useState } from "react";
import type { SkillDefinition } from "../types/haxen";
import type { AgentDIDInfo, SkillDIDInfo } from "../types/did";
import { DIDIdentityBadge, DIDStatusBadge } from "./did/DIDStatusBadge";
import { useDIDNotifications } from "./ui/notification";

interface SkillsListProps {
  skills: SkillDefinition[];
  didInfo?: AgentDIDInfo;
  nodeId?: string;
  showDIDDetails?: boolean;
  compact?: boolean;
}

interface SkillItemProps {
  skill: SkillDefinition;
  skillDID?: SkillDIDInfo;
  showDIDDetails?: boolean;
  compact?: boolean;
  onCopyDID?: (did: string) => void;
}

const SkillItem: React.FC<SkillItemProps> = ({
  skill,
  skillDID,
  showDIDDetails = true,
  compact = false,
  onCopyDID,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (compact) {
    return (
      <div className="flex items-center gap-2 p-2 border rounded-lg bg-white hover:bg-gray-50 transition-colors">
        <Badge
          variant="outline"
          className="text-xs bg-purple-50 text-purple-700 border-purple-200"
        >
          {skill.id}
        </Badge>

        {skillDID && showDIDDetails && (
          <div className="flex items-center gap-1">
            <DIDStatusBadge status="active" size="sm" />
            <DIDIdentityBadge
              did={skillDID.did}
              maxLength={20}
              showCopyButton={false}
            />
          </div>
        )}

        {skill.tags && skill.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {skill.tags.slice(0, 2).map((tag) => (
              <Badge
                key={tag}
                variant="secondary"
                className="text-xs opacity-70"
              >
                {tag}
              </Badge>
            ))}
            {skill.tags.length > 2 && (
              <Badge variant="secondary" className="text-xs opacity-50">
                +{skill.tags.length - 2}
              </Badge>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="space-y-3">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className="bg-purple-50 text-purple-700 border-purple-200"
              >
                {skill.id}
              </Badge>
              {skillDID && showDIDDetails && (
                <DIDStatusBadge status="active" size="sm" />
              )}
            </div>

            {skillDID && showDIDDetails && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsExpanded(!isExpanded)}
                className="h-6 w-6 p-0"
              >
                <Information className="h-3 w-3" />
              </Button>
            )}
          </div>

          {/* DID Information */}
          {skillDID && showDIDDetails && (
            <div className="space-y-2">
              <DIDIdentityBadge
                did={skillDID.did}
                maxLength={35}
                onCopy={onCopyDID}
              />

              {isExpanded && (
                <div className="space-y-2 p-3 bg-gray-50 rounded-lg text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Exposure Level:</span>
                    <Badge variant="outline" className="text-xs">
                      {skillDID.exposure_level}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Created:</span>
                    <span className="text-gray-800">
                      {new Date(skillDID.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  {skillDID.tags.length > 0 && (
                    <div>
                      <span className="text-gray-600">DID Tags:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {skillDID.tags.map((tag, index) => (
                          <Badge
                            key={index}
                            variant="secondary"
                            className="text-xs"
                          >
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Skill Tags */}
          {skill.tags && skill.tags.length > 0 && (
            <div>
              <div className="text-xs text-gray-600 mb-1">Skill Tags:</div>
              <div className="flex flex-wrap gap-1">
                {skill.tags.map((tag) => (
                  <Badge
                    key={tag}
                    variant="secondary"
                    className="text-xs opacity-70"
                  >
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

const SkillsList: React.FC<SkillsListProps> = ({
  skills,
  didInfo,
  showDIDDetails = true,
  compact = false,
}) => {
  const { didCopied } = useDIDNotifications();

  const handleCopyDID = (did: string) => {
    navigator.clipboard.writeText(did);
    didCopied("Skill DID");
  };

  if (!skills || skills.length === 0) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Function className="h-4 w-4 text-muted-foreground" />
          <h4 className="text-sm font-medium">Skills (0)</h4>
        </div>
        <p className="text-body-small">No skills available.</p>
      </div>
    );
  }

  // Count skills with DIDs
  const skillsWithDIDs = didInfo
    ? skills.filter((skill) => didInfo.skills[skill.id]).length
    : 0;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Function className="h-4 w-4 text-muted-foreground" />
          <h4 className="text-sm font-medium">Skills ({skills.length})</h4>
          {showDIDDetails && didInfo && skillsWithDIDs > 0 && (
            <Badge
              variant="outline"
              className="text-xs bg-green-50 text-green-700 border-green-200"
            >
              {skillsWithDIDs} with DID
            </Badge>
          )}
        </div>
      </div>

      <div
        className={
          compact
            ? "space-y-2"
            : "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3"
        }
      >
        {skills.map((skill) => {
          const skillDID = didInfo?.skills[skill.id];

          return (
            <SkillItem
              key={skill.id}
              skill={skill}
              skillDID={skillDID}
              showDIDDetails={showDIDDetails}
              compact={compact}
              onCopyDID={handleCopyDID}
            />
          );
        })}
      </div>
    </div>
  );
};

export default SkillsList;
