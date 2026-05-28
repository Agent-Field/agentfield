import { Checkmark, ChevronDown, Filter } from "@/components/ui/icon-bridge";
import { useState, useRef, useEffect } from "react";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { TagBadge } from "./TagBadge";

interface TagFilterProps {
  availableTags: string[];
  selectedTags: string[];
  onTagsChange: (tags: string[]) => void;
  placeholder?: string;
  className?: string;
}

export function TagFilter({
  availableTags,
  selectedTags,
  onTagsChange,
  placeholder = "Filter by tags...",
  className = "",
}: TagFilterProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Filter available tags based on search term
  const filteredTags = availableTags.filter(tag =>
    tag.toLowerCase().includes(searchTerm.toLowerCase()) &&
    !selectedTags.includes(tag)
  );

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setSearchTerm("");
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Focus input when dropdown opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleTagSelect = (tag: string) => {
    if (!selectedTags.includes(tag)) {
      onTagsChange([...selectedTags, tag]);
    }
    setSearchTerm("");
  };

  const handleTagRemove = (tag: string) => {
    onTagsChange(selectedTags.filter(t => t !== tag));
  };

  const handleClearAll = () => {
    onTagsChange([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setIsOpen(false);
      setSearchTerm("");
    } else if (e.key === "Enter" && filteredTags.length > 0) {
      e.preventDefault();
      handleTagSelect(filteredTags[0]);
    }
  };

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      {/* Selected Tags Display */}
      {selectedTags.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {selectedTags.map(tag => (
            <TagBadge
              key={tag}
              tag={tag}
              removable
              onRemove={handleTagRemove}
            />
          ))}
          {selectedTags.length > 1 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearAll}
              className="h-6 px-2 text-xs text-muted-foreground hover:text-muted-foreground"
            >
              Clear all
            </Button>
          )}
        </div>
      )}

      {/* Filter Trigger */}
      <Button
        variant="outline"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full justify-between text-left font-normal"
        disabled={availableTags.length === 0}
      >
        <div className="flex items-center gap-2">
          <Filter size={16} className="text-muted-foreground" />
          <span className="text-muted-foreground">
            {selectedTags.length > 0
              ? `${selectedTags.length} tag${selectedTags.length > 1 ? 's' : ''} selected`
              : placeholder
            }
          </span>
        </div>
        <ChevronDown size={16} className={`transition-transform duration-fast ${isOpen ? 'rotate-180' : ''}`} />
      </Button>

      {/* Dropdown */}
      {isOpen && (
        <Card className="absolute top-full left-0 right-0 z-50 mt-1 shadow-lg border">
          <CardContent className="p-0">
            {/* Search Input */}
            <div className="p-3 border-b">
              <input
                ref={inputRef}
                type="text"
                placeholder="Search tags..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={handleKeyDown}
                className="w-full px-3 py-2 text-sm bg-muted border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-accent-primary focus:border-transparent text-foreground"
              />
            </div>

            {/* Available Tags List */}
            <div className="max-h-48 overflow-y-auto">
              {filteredTags.length > 0 ? (
                <div className="p-2">
                  {filteredTags.map(tag => (
                    <button
                      key={tag}
                      onClick={() => handleTagSelect(tag)}
                      className="w-full flex items-center justify-between px-3 py-2 text-sm hover:bg-accent rounded-md transition-colors text-foreground"
                    >
                      <span>{tag}</span>
                      <Checkmark size={16} className="opacity-0" />
                    </button>
                  ))}
                </div>
              ) : (
                <div className="p-4 text-center text-sm text-muted-foreground">
                  {searchTerm ? "No tags found" : "All tags selected"}
                </div>
              )}
            </div>

            {/* Footer */}
            {availableTags.length > 0 && (
              <div className="p-3 border-t bg-muted/30">
                <div className="text-xs text-muted-foreground">
                  {availableTags.length} total tag{availableTags.length > 1 ? 's' : ''} available
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
