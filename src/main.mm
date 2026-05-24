#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>
#import "../include/Renderer.hpp"
#import "../include/AppDelegate.hpp"

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    NSApplication *app = [NSApplication sharedApplication];
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];

    NSWindow *window =
        [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 1600, 900)
                                    styleMask:(NSWindowStyleMaskTitled |
                                               NSWindowStyleMaskClosable)
                                      backing:NSBackingStoreBuffered // back and front buffer rendering scheme
                                        defer:NO]; // creates the window immediately
    [window setTitle:@"Image Liquifier (drag and drop image into window!)"];
    [window makeKeyAndOrderFront:nil];
    [window center];

    AppDelegate *delegate = [[AppDelegate alloc] init];
    [app setDelegate:delegate];
    [window setDelegate:delegate];

    id<MTLDevice> device = MTLCreateSystemDefaultDevice(); // get default GPU
    SimView *view = [[SimView alloc] initWithFrame:window.contentView.bounds device:device];
    view.clearColor = MTLClearColorMake(0.1, 0.1, 0.1, 1.0);
    view.preferredFramesPerSecond = 60; // Hint for VSync
    [window setContentView:view];
    [window setAcceptsMouseMovedEvents:YES];

    Renderer *renderer = [[Renderer alloc] initWithMetalKitView:view];
    view.delegate = renderer;

    [app run];
  }
  return 0;
}
