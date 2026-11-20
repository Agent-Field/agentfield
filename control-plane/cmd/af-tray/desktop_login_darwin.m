#import <Cocoa/Cocoa.h>
#import <ServiceManagement/ServiceManagement.h>
#include <string.h>

static NSString *const loginPlist = @"ai.agentfield.tray.login.plist";

int af_desktop_login_status(void) {
    @autoreleasepool {
        if (@available(macOS 13.0, *)) {
            NSString *path = [NSBundle.mainBundle.bundlePath
                stringByAppendingPathComponent:[@"Contents/Library/LaunchAgents" stringByAppendingPathComponent:loginPlist]];
            if ([NSFileManager.defaultManager fileExistsAtPath:path]) {
                return (int)[SMAppService agentServiceWithPlistName:loginPlist].status;
            }
        }
        return -1;
    }
}

char *af_set_desktop_login(int enable) {
    @autoreleasepool {
        if (@available(macOS 13.0, *)) {
            SMAppService *service = [SMAppService agentServiceWithPlistName:loginPlist];
            NSError *error = nil;
            BOOL ok = enable ? [service registerAndReturnError:&error] : [service unregisterAndReturnError:&error];
            if (!ok && !(enable && service.status == SMAppServiceStatusRequiresApproval)) {
                return strdup(error.localizedDescription.UTF8String ?: "unknown macOS error");
            }
        }
        return NULL;
    }
}

void af_accessory_app(void) {
    [NSApplication.sharedApplication setActivationPolicy:NSApplicationActivationPolicyAccessory];
}
