//go:build darwin

package main

/*
#cgo CFLAGS: -fobjc-arc
#cgo LDFLAGS: -framework Cocoa
#include <stdlib.h>

char *af_prompt_secret(const char *title, const char *message);
void af_show_alert(const char *title, const char *message);
*/
import "C"

import "unsafe"

// promptSecret shows a native alert with a masked text field. It returns
// ok=false when the user cancels.
func promptSecret(title, message string) (string, bool) {
	t, m := C.CString(title), C.CString(message)
	defer C.free(unsafe.Pointer(t))
	defer C.free(unsafe.Pointer(m))
	out := C.af_prompt_secret(t, m)
	if out == nil {
		return "", false
	}
	defer C.free(unsafe.Pointer(out))
	return C.GoString(out), true
}

// showAlert shows a native one-button alert and returns once it is dismissed.
func showAlert(title, message string) {
	t, m := C.CString(title), C.CString(message)
	defer C.free(unsafe.Pointer(t))
	defer C.free(unsafe.Pointer(m))
	C.af_show_alert(t, m)
}
