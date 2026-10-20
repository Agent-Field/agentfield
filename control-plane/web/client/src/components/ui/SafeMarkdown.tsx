import { Component, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { safeExternalUrl } from "@/utils/safeExternalUrl";

interface MarkdownNode {
  type: string;
  value?: string;
  children?: MarkdownNode[];
}

/** Preserve single line breaks in all prose, including tight list items. */
function remarkLineBreaks() {
  return function transform(tree: MarkdownNode) {
    const stack = [tree];
    while (stack.length > 0) {
      const node = stack.pop()!;
      if (node.type === "code" || node.type === "inlineCode" || !node.children) continue;
      node.children = node.children.flatMap((child): MarkdownNode[] => {
        if (child.type === "text" && child.value?.includes("\n")) {
          return child.value.split("\n").flatMap((value, index) =>
            index === 0 ? [{ type: "text", value }] : [{ type: "break" }, { type: "text", value }],
          );
        }
        stack.push(child);
        return [child];
      });
    }
  };
}

interface MarkdownBoundaryProps {
  content: string;
  children: ReactNode;
}

interface MarkdownBoundaryState {
  content: string;
  hasError: boolean;
}

/** Keep parser and renderer failures local to this preview. */
class MarkdownBoundary extends Component<MarkdownBoundaryProps, MarkdownBoundaryState> {
  state: MarkdownBoundaryState = { content: this.props.content, hasError: false };

  static getDerivedStateFromProps(props: MarkdownBoundaryProps, state: MarkdownBoundaryState) {
    return props.content === state.content ? null : { content: props.content, hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    return this.state.hasError ? (
      <pre className="whitespace-pre-wrap break-words text-sm">{this.props.content}</pre>
    ) : this.props.children;
  }
}

/** Untrusted markdown: raw HTML stays visible text and images never load. */
export function SafeMarkdown({ content, className }: { content: string; className?: string }) {
  return (
    <div className={cn("prose prose-sm max-w-none text-foreground", className)}>
      <MarkdownBoundary content={content}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkLineBreaks]}
          urlTransform={(url) => safeExternalUrl(url, true) ?? ""}
          components={{
            a: ({ href, title, children }) => href ? (
              <a href={href} title={title} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
                {children}
              </a>
            ) : <>{children}</>,
            img: ({ alt }) => <>{alt}</>,
            h1: ({ children }) => <h1 className="text-xl font-semibold mb-3">{children}</h1>,
            h2: ({ children }) => <h2 className="text-base font-semibold mb-2">{children}</h2>,
            p: ({ children }) => <p className="mt-2">{children}</p>,
            ul: ({ children }) => <ul className="list-disc list-inside my-2">{children}</ul>,
            ol: ({ children, start }) => <ol start={start} className="list-decimal list-inside my-2">{children}</ol>,
            code: ({ children, className }) => <code className={cn("bg-muted px-1 rounded text-sm font-mono", className)}>{children}</code>,
            pre: ({ children }) => <pre className="bg-muted p-3 rounded-md overflow-auto my-2">{children}</pre>,
            blockquote: ({ children }) => <blockquote className="border-l-4 border-border pl-3 italic my-2">{children}</blockquote>,
          }}
        >
          {content}
        </ReactMarkdown>
      </MarkdownBoundary>
    </div>
  );
}
