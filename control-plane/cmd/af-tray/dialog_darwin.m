#import <Cocoa/Cocoa.h>
#include <string.h>

// Native dialogs for the tray. They replace `osascript -e 'display dialog …
// with hidden answer'`, which is also exactly how credential-phishing malware
// asks for a password, and which endpoint-security tools therefore flag.
// An NSAlert from the signed helper itself is attributable to AgentField.
//
// The systray run loop owns the main thread; these are called from a goroutine,
// so they hop onto the main queue and block until the alert is dismissed.

static void runOnMain(void (^block)(void)) {
    if ([NSThread isMainThread]) {
        block();
    } else {
        dispatch_sync(dispatch_get_main_queue(), block);
    }
}

// af_prompt_secret asks for a secret in a masked field. It returns a malloc'd
// UTF-8 string the caller frees, or NULL when the user cancels.
char *af_prompt_secret(const char *title, const char *message) {
    __block char *result = NULL;
    NSString *t = [NSString stringWithUTF8String:title ?: ""];
    NSString *m = [NSString stringWithUTF8String:message ?: ""];
    runOnMain(^{
        @autoreleasepool {
            NSAlert *alert = [[NSAlert alloc] init];
            alert.messageText = t;
            alert.informativeText = m;
            [alert addButtonWithTitle:@"Save"];
            [alert addButtonWithTitle:@"Cancel"];
            NSSecureTextField *field = [[NSSecureTextField alloc] initWithFrame:NSMakeRect(0, 0, 320, 24)];
            alert.accessoryView = field;
            alert.window.initialFirstResponder = field;
            [NSApp activateIgnoringOtherApps:YES];
            if ([alert runModal] == NSAlertFirstButtonReturn) {
                result = strdup(field.stringValue.UTF8String ?: "");
            }
        }
    });
    return result;
}

// af_show_alert shows a one-button informational alert and returns when it is
// dismissed.
void af_show_alert(const char *title, const char *message) {
    NSString *t = [NSString stringWithUTF8String:title ?: ""];
    NSString *m = [NSString stringWithUTF8String:message ?: ""];
    runOnMain(^{
        @autoreleasepool {
            NSAlert *alert = [[NSAlert alloc] init];
            alert.alertStyle = NSAlertStyleWarning;
            alert.messageText = t;
            alert.informativeText = m;
            [alert addButtonWithTitle:@"OK"];
            [NSApp activateIgnoringOtherApps:YES];
            [alert runModal];
        }
    });
}
