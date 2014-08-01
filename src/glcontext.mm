#include "glcontext.h"
#if defined(__OBJC__)
#import <Cocoa/Cocoa.h>
#else
#include <ApplicationServices/ApplicationServices.h>
typedef void* id;
#endif

bool _HasNSGLContext()
{
		NSOpenGLContext* context = [NSOpenGLContext currentContext];
		if (context) {
		    return true;
		} else {
		    return false;
		}
}
