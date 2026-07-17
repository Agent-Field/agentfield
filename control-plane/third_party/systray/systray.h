#include "stdbool.h"

extern void systray_ready();
extern void systray_on_exit();
extern void systray_left_click();
extern void systray_right_click();
extern void systray_menu_item_selected(int menu_id);
extern void systray_menu_will_open();
void registerSystray(void);
void nativeEnd(void);
int nativeLoop(void);
void nativeStart(void);

void setIcon(const char* iconBytes, int length, bool template);
void setMenuItemIcon(const char* iconBytes, int length, int menuId, bool template);
// AgentField patch: set a non-template menu-item image at an explicit point size
// (width/height in points; PNG pixels are expected to be 2x for retina). Unlike
// setMenuItemIcon this never clamps to 16x16 and never marks the image template,
// so wide colored charts/bars can be shown. See third_party/systray/PATCHES.md.
void setMenuItemImage(const char* iconBytes, int length, int menuId, int widthPt, int heightPt);
// AgentField patch #2: set the status-bar button's image at an explicit point
// size (width/height in points; PNG pixels are expected to be 2x for retina),
// non-template so it keeps its own colors. Unlike setIcon this never clamps to
// 16x16 and never marks the image template, so a wide colored menu-bar widget
// (brand badge + sparkline) can be shown next to the title. It reuses the
// existing setIcon: selector. See third_party/systray/PATCHES.md.
void setStatusImage(const char* iconBytes, int length, int widthPt, int heightPt);
void setTitle(char* title);
void setTooltip(char* tooltip);
void setRemovalAllowed(bool allowed);
void add_or_update_menu_item(int menuId, int parentMenuId, char* title, char* tooltip, short disabled, short checked, short isCheckable);
void add_separator(int menuId, int parentId);
void hide_menu_item(int menuId);
void remove_menu_item(int menuId);
void show_menu_item(int menuId);
void reset_menu();
void show_menu();
void quit();
