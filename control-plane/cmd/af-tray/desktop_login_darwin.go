//go:build darwin

package main

/*
#cgo LDFLAGS: -framework Cocoa -framework ServiceManagement
#include <stdlib.h>

int af_desktop_login_status(void);
char *af_set_desktop_login(int enable);
void af_accessory_app(void);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type loginStatus int

const (
	desktopLoginUnavailable loginStatus = -1
	desktopLoginEnabled     loginStatus = 1
	desktopLoginApproval    loginStatus = 2
)

func desktopLoginStatus() loginStatus {
	return loginStatus(C.af_desktop_login_status())
}

func (status loginStatus) registered() bool {
	return status == desktopLoginEnabled || status == desktopLoginApproval
}

func setDesktopLogin(status loginStatus, enable bool) error {
	if status == desktopLoginUnavailable || status.registered() == enable {
		return nil
	}
	var value C.int
	if enable {
		value = 1
	}
	if message := C.af_set_desktop_login(value); message != nil {
		defer C.free(unsafe.Pointer(message))
		return fmt.Errorf("configure tray login item: %s", C.GoString(message))
	}
	return nil
}

func setupTrayApp() { C.af_accessory_app() }
