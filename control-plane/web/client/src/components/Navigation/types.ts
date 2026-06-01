import type { IconProps } from '../ui/icon';

export interface NavigationItem {
  id: string;
  label: string;
  href: string;
  icon?: IconProps['name'];
  description?: string;
  disabled?: boolean;
}

export interface NavigationSection {
  id: string;
  title: string;
  items: NavigationItem[];
}
