import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Copy,
  List,
  Document,
  Maximize
} from '@/components/ui/icon-bridge';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';
import { SmartStringRenderer } from './SmartStringRenderer';
import { JsonModal } from './JsonModal';

interface EnhancedJsonViewerProps {
  data: unknown;
  title?: string;
  className?: string;
  maxInlineHeight?: number;
}

type JsonValueType =
  | 'string'
  | 'number'
  | 'boolean'
  | 'array'
  | 'object'
  | 'null'
  | 'undefined';

interface JsonItem {
  key: string;
  value: unknown;
  type: JsonValueType;
  path: string[];
  isExpandable: boolean;
}

export function EnhancedJsonViewer({
  data,
  className = "",
  maxInlineHeight = 200
}: EnhancedJsonViewerProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [modalState, setModalState] = useState<{
    isOpen: boolean;
    content: unknown;
    path: string[];
    title: string;
  }>({
    isOpen: false,
    content: null,
    path: [],
    title: ''
  });

  const toggleExpanded = (itemKey: string) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(itemKey)) {
      newExpanded.delete(itemKey);
    } else {
      newExpanded.add(itemKey);
    }
    setExpandedItems(newExpanded);
  };

  const openModal = (content: unknown, path: string[], itemTitle: string) => {
    setModalState({
      isOpen: true,
      content,
      path,
      title: itemTitle
    });
  };

  const closeModal = () => {
    setModalState({
      isOpen: false,
      content: null,
      path: [],
      title: ''
    });
  };

  const copyToClipboard = (content: unknown) => {
    const text = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
    navigator.clipboard.writeText(text);
  };

  const getJsonValueType = (value: unknown): JsonValueType => {
    if (value === null) return 'null';
    if (value === undefined) return 'undefined';
    if (Array.isArray(value)) return 'array';
    if (typeof value === 'string') return 'string';
    if (typeof value === 'number') return 'number';
    if (typeof value === 'boolean') return 'boolean';
    return 'object';
  };

  const processJsonData = (obj: unknown, parentPath: string[] = []): JsonItem[] => {
    if (obj === null || obj === undefined) {
      return [];
    }

    if (typeof obj !== 'object') {
      return [{
        key: 'value',
        value: obj,
        type: getJsonValueType(obj),
        path: parentPath,
        isExpandable: false
      }];
    }

    if (Array.isArray(obj)) {
      return [{
        key: 'array',
        value: obj,
        type: 'array',
        path: parentPath,
        isExpandable: obj.length > 0
      }];
    }

    return Object.entries(obj).map(([key, value]) => {
      const path = [...parentPath, key];
      const type = getJsonValueType(value);

      return {
        key,
        value,
        type,
        path,
        isExpandable: type === 'object' || type === 'array'
      };
    });
  };

  const formatLabel = (key: string): string => {
    return key
      .replace(/([A-Z])/g, ' $1')
      .replace(/[_-]/g, ' ')
      .replace(/\b\w/g, (match) => match.toUpperCase())
      .trim();
  };

  const renderValue = (item: JsonItem) => {
    const itemKey = item.path.join('.');
    const isExpanded = expandedItems.has(itemKey);

    switch (item.type) {
      case 'string':
        {
          const stringValue = typeof item.value === 'string' ? item.value : String(item.value);
        return (
          <SmartStringRenderer
            content={stringValue}
            label={item.key}
            path={item.path}
            onOpenModal={() => openModal(stringValue, item.path, formatLabel(item.key))}
            maxInlineHeight={maxInlineHeight}
          />
        );
        }

      case 'number':
        {
          const numberValue = typeof item.value === 'number' ? item.value : Number(item.value);
        return (
          <div className="flex items-center gap-2">
            <span className="text-sm font-mono text-foreground">
              {numberValue.toLocaleString()}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => copyToClipboard(item.value)}
              className="h-6 w-6 p-0"
            >
              <Copy className="h-3 w-3" />
            </Button>
          </div>
        );
        }

      case 'boolean':
        {
          const booleanValue = Boolean(item.value);
        return (
          <div className="flex items-center gap-2">
            <Badge variant={booleanValue ? "default" : "secondary"} className="text-xs">
              {booleanValue ? 'true' : 'false'}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => copyToClipboard(item.value)}
              className="h-6 w-6 p-0"
            >
              <Copy className="h-3 w-3" />
            </Button>
          </div>
        );
        }

      case 'null':
        return (
          <div className="flex items-center gap-2">
            <span className="text-body-small italic">null</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => copyToClipboard(null)}
              className="h-6 w-6 p-0"
            >
              <Copy className="h-3 w-3" />
            </Button>
          </div>
        );

      case 'array':
        {
          const arrayValue = Array.isArray(item.value) ? item.value : [];
          return (
            <div className="space-y-2">
              <Collapsible open={isExpanded} onOpenChange={() => toggleExpanded(itemKey)}>
                <div className="flex items-center gap-2">
                  <CollapsibleTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                    >
                      {isExpanded ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronRight className="h-3 w-3" />
                      )}
                    </Button>
                  </CollapsibleTrigger>

                  <div className="flex items-center gap-2 flex-1">
                    <List className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-foreground">
                      Array
                    </span>
                    <Badge variant="secondary" className="text-xs">
                      {arrayValue.length} items
                    </Badge>
                  </div>

                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(arrayValue)}
                      className="h-6 w-6 p-0"
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => openModal(arrayValue, item.path, formatLabel(item.key))}
                      className="h-6 w-6 p-0"
                    >
                      <Maximize className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                <CollapsibleContent>
                  <div className="ml-6 mt-2 space-y-2">
                    {arrayValue.slice(0, 10).map((arrayItem: unknown, index: number) => (
                      <div key={index} className="flex items-start gap-2 p-2 bg-muted/30 rounded text-sm">
                        <span className="text-muted-foreground font-mono text-xs mt-0.5 flex-shrink-0">
                          [{index}]
                        </span>
                        <div className="flex-1 min-w-0">
                          {typeof arrayItem === 'string' ? (
                            <SmartStringRenderer
                              content={arrayItem}
                              label={`${item.key}[${index}]`}
                              path={[...item.path, index.toString()]}
                              onOpenModal={() => openModal(arrayItem, [...item.path, index.toString()], `${formatLabel(item.key)}[${index}]`)}
                              maxInlineHeight={100}
                            />
                          ) : typeof arrayItem === 'object' && arrayItem !== null ? (
                            <div className="text-xs">
                              <pre className="text-foreground whitespace-pre-wrap break-words overflow-hidden">
                                {JSON.stringify(arrayItem, null, 2)}
                              </pre>
                            </div>
                          ) : (
                            <span className="text-foreground">
                              {String(arrayItem)}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                    {arrayValue.length > 10 && (
                      <div className="text-body-small text-center py-2">
                        ... and {arrayValue.length - 10} more items
                      </div>
                    )}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </div>
          );
        }

      case 'undefined':
        return (
          <div className="flex items-center gap-2">
            <span className="text-body-small italic">undefined</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => copyToClipboard(undefined)}
              className="h-6 w-6 p-0"
            >
              <Copy className="h-3 w-3" />
            </Button>
          </div>
        );

      case 'object':
        {
          const objectValue =
            item.value && typeof item.value === 'object' && !Array.isArray(item.value)
              ? item.value
              : {};
        return (
          <div className="space-y-2">
            <Collapsible open={isExpanded} onOpenChange={() => toggleExpanded(itemKey)}>
              <div className="flex items-center gap-2">
                <CollapsibleTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0"
                  >
                    {isExpanded ? (
                      <ChevronDown className="h-3 w-3" />
                    ) : (
                      <ChevronRight className="h-3 w-3" />
                    )}
                  </Button>
                </CollapsibleTrigger>

                <div className="flex items-center gap-2 flex-1">
                  <Document className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-foreground">
                    Object
                  </span>
                  <Badge variant="secondary" className="text-xs">
                    {Object.keys(objectValue).length} keys
                  </Badge>
                </div>

                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(objectValue)}
                    className="h-6 w-6 p-0"
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => openModal(objectValue, item.path, formatLabel(item.key))}
                    className="h-6 w-6 p-0"
                  >
                    <Maximize className="h-3 w-3" />
                  </Button>
                </div>
              </div>

              <CollapsibleContent>
                <div className="ml-6 mt-2">
                  <EnhancedJsonViewer
                    data={objectValue}
                    maxInlineHeight={maxInlineHeight}
                  />
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>
        );
        }

      default:
        return (
          <div className="flex items-center gap-2">
            <span className="text-sm text-foreground">
              {String(item.value)}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => copyToClipboard(item.value)}
              className="h-6 w-6 p-0"
            >
              <Copy className="h-3 w-3" />
            </Button>
          </div>
        );
    }
  };

  const items = processJsonData(data);

  if (items.length === 0) {
    return (
      <div className={`text-center py-8 text-muted-foreground ${className}`}>
        <Document className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p>No data to display</p>
      </div>
    );
  }

  return (
    <div className={`space-y-3 ${className}`}>
      {items.map((item, index) => (
        <div key={`${item.key}-${index}`} className="space-y-2">
          {/* Key-Value Row */}
          <div className="flex items-start gap-3 py-2">
            {/* Key */}
            <div className="flex-shrink-0 w-32 sm:w-40 min-w-0">
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-sm font-medium text-foreground truncate block min-w-0" title={formatLabel(item.key)}>
                  {formatLabel(item.key)}
                </span>
                {item.type !== 'string' && item.type !== 'number' && item.type !== 'boolean' && item.type !== 'null' && item.type !== 'undefined' && (
                  <Badge variant="outline" className="text-xs flex-shrink-0">
                    {item.type}
                  </Badge>
                )}
              </div>
            </div>

            {/* Value */}
            <div className="flex-1 min-w-0">
              {renderValue(item)}
            </div>
          </div>

          {/* Separator */}
          {index < items.length - 1 && (
            <div className="border-b border-border/50" />
          )}
        </div>
      ))}

      {/* Modal */}
      <JsonModal
        isOpen={modalState.isOpen}
        onClose={closeModal}
        content={modalState.content}
        path={modalState.path}
        title={modalState.title}
      />
    </div>
  );
}
