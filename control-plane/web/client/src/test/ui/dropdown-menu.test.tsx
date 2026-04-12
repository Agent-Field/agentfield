import { fireEvent, render, screen } from "@testing-library/react";
import {
  forwardRef,
  type ButtonHTMLAttributes,
  type HTMLAttributes,
  type ReactNode,
} from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@radix-ui/react-dropdown-menu", () => {
  const Root = ({ children }: { children: ReactNode }) => <div>{children}</div>;
  const Trigger = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement>>(
    ({ children, ...props }, ref) => (
      <button ref={ref} type="button" {...props}>
        {children}
      </button>
    ),
  );
  const Content = forwardRef<
    HTMLDivElement,
    HTMLAttributes<HTMLDivElement> & { sideOffset?: number }
  >(({ children, sideOffset: _sideOffset, ...props }, ref) => (
      <div ref={ref} {...props}>
        {children}
      </div>
    ));
  const Item = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement>>(
    ({ children, onClick, ...props }, ref) => (
      <button ref={ref} type="button" onClick={onClick} {...props}>
        {children}
      </button>
    ),
  );
  const CheckboxItem = forwardRef<
    HTMLButtonElement,
    ButtonHTMLAttributes<HTMLButtonElement> & {
      checked?: boolean;
      onCheckedChange?: (checked: boolean) => void;
    }
  >(({ children, checked, onCheckedChange, ...props }, ref) => (
    <button
      ref={ref}
      type="button"
      role="menuitemcheckbox"
      aria-checked={checked}
      onClick={() => onCheckedChange?.(!checked)}
      {...props}
    >
      {children}
    </button>
  ));
  const RadioGroup = ({
    children,
    value,
    onValueChange,
  }: {
    children: ReactNode;
    value?: string;
    onValueChange?: (value: string) => void;
  }) => <div data-value={value} data-on-value-change={Boolean(onValueChange)}>{children}</div>;
  const RadioItem = forwardRef<
    HTMLButtonElement,
    ButtonHTMLAttributes<HTMLButtonElement> & { value: string }
  >(({ children, value, onClick, ...props }, ref) => (
    <button ref={ref} type="button" role="menuitemradio" data-value={value} onClick={onClick} {...props}>
      {children}
    </button>
  ));
  const Label = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
    ({ children, ...props }, ref) => (
      <div ref={ref} {...props}>
        {children}
      </div>
    ),
  );
  const Separator = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>((props, ref) => (
    <div ref={ref} {...props} />
  ));
  const SubTrigger = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement>>(
    ({ children, ...props }, ref) => (
      <button ref={ref} type="button" {...props}>
        {children}
      </button>
    ),
  );
  const SubContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
    ({ children, ...props }, ref) => (
      <div ref={ref} {...props}>
        {children}
      </div>
    ),
  );
  const Group = ({ children }: { children: ReactNode }) => <div>{children}</div>;
  const Portal = ({ children }: { children: ReactNode }) => <div>{children}</div>;
  const Sub = ({ children }: { children: ReactNode }) => <div>{children}</div>;
  const ItemIndicator = ({ children }: { children: ReactNode }) => <span>{children}</span>;

  function withName<T extends { displayName?: string }>(component: T, name: string): T {
    component.displayName = name;
    return component;
  }

  return {
    Root,
    Trigger: withName(Trigger, "Trigger"),
    Group,
    Portal,
    Sub,
    RadioGroup,
    SubTrigger: withName(SubTrigger, "SubTrigger"),
    SubContent: withName(SubContent, "SubContent"),
    Content: withName(Content, "Content"),
    Item: withName(Item, "Item"),
    CheckboxItem: withName(CheckboxItem, "CheckboxItem"),
    RadioItem: withName(RadioItem, "RadioItem"),
    Label: withName(Label, "Label"),
    Separator: withName(Separator, "Separator"),
    ItemIndicator,
  };
});

import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

describe("dropdown-menu", () => {
  it("renders content, label, separator, shortcut, and item click handlers", async () => {
    const onSelect = vi.fn();

    const { container } = render(
      <DropdownMenu>
        <DropdownMenuTrigger>Open menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuLabel inset>Actions</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuGroup>
            <DropdownMenuItem inset onClick={onSelect}>
              Open
              <DropdownMenuShortcut>Cmd+O</DropdownMenuShortcut>
            </DropdownMenuItem>
          </DropdownMenuGroup>
          <DropdownMenuSub open>
            <DropdownMenuSubTrigger inset>More</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuItem>Nested item</DropdownMenuItem>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        </DropdownMenuContent>
      </DropdownMenu>,
    );

    expect(screen.getByText("Actions")).toBeInTheDocument();
    expect(screen.getByText("Nested item")).toBeInTheDocument();
    expect(screen.getByText("Cmd+O")).toBeInTheDocument();
    expect(container.querySelector(".pl-8")).not.toBeNull();
    expect(container.querySelector(".origin-\\[--radix-dropdown-menu-content-transform-origin\\]")).not.toBeNull();

    fireEvent.click(screen.getAllByRole("button", { name: /open/i })[1]);
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it("supports checkbox and radio item selection", async () => {
    const onCheckedChange = vi.fn();
    const onRadioClick = vi.fn();

    render(
      <DropdownMenu>
        <DropdownMenuTrigger>Toggle menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuCheckboxItem checked onCheckedChange={onCheckedChange}>
            Show logs
          </DropdownMenuCheckboxItem>
          <DropdownMenuRadioGroup value="desc" onValueChange={vi.fn()}>
            <DropdownMenuRadioItem value="asc" onClick={onRadioClick}>
              Ascending
            </DropdownMenuRadioItem>
            <DropdownMenuRadioItem value="desc">Descending</DropdownMenuRadioItem>
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>,
    );

    const checkbox = screen.getByRole("menuitemcheckbox", { name: "Show logs" });
    const radio = screen.getByRole("menuitemradio", { name: "Ascending" });

    fireEvent.click(checkbox);
    fireEvent.click(radio);

    expect(onCheckedChange).toHaveBeenCalledWith(false);
    expect(onRadioClick).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("menuitemradio", { name: "Descending" })).toHaveAttribute("data-value", "desc");
  });
});
