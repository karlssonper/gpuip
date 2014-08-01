#if defined(__OBJC__)
#import <Cocoa/Cocoa.h>
#else
#error "No objective C found"
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
