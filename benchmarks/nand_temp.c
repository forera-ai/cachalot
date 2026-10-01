/* NAND temperature of the internal SSD from the HID sensor "NAND CH0 temp", one line per period: epoch seconds, name=degrees C.
 * smartctl is not installed. HANDOFF section 18.46.
 *   clang -o /tmp/nand_temp benchmarks/nand_temp.c -framework CoreFoundation -framework IOKit && /tmp/nand_temp 5
 * A period of 0 prints one line and exits. */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <CoreFoundation/CoreFoundation.h>
typedef struct __IOHIDEventSystemClient *IOHIDEventSystemClientRef;
typedef struct __IOHIDServiceClient *IOHIDServiceClientRef;
typedef struct __IOHIDEvent *IOHIDEventRef;
extern IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef);
extern void IOHIDEventSystemClientSetMatching(IOHIDEventSystemClientRef, CFDictionaryRef);
extern CFArrayRef IOHIDEventSystemClientCopyServices(IOHIDEventSystemClientRef);
extern CFTypeRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef, CFStringRef);
extern IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef, int64_t, int32_t, int64_t);
extern double IOHIDEventGetFloatValue(IOHIDEventRef, int32_t);
#define kIOHIDEventTypeTemperature 15
int main(int argc, char **argv) {
  int period = argc > 1 ? atoi(argv[1]) : 5;
  int page = 0xff00, usage = 5;
  CFNumberRef p = CFNumberCreate(NULL, kCFNumberIntType, &page), u = CFNumberCreate(NULL, kCFNumberIntType, &usage);
  const void *k[2] = {CFSTR("PrimaryUsagePage"), CFSTR("PrimaryUsage")}, *v[2] = {p, u};
  CFDictionaryRef m = CFDictionaryCreate(NULL, k, v, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  IOHIDEventSystemClientRef c = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
  IOHIDEventSystemClientSetMatching(c, m);
  for (;;) {
    CFArrayRef a = IOHIDEventSystemClientCopyServices(c);
    printf("%ld", time(NULL));
    for (CFIndex i = 0; a && i < CFArrayGetCount(a); i++) {
      IOHIDServiceClientRef s = (IOHIDServiceClientRef)CFArrayGetValueAtIndex(a, i);
      char name[128] = "?";
      CFStringRef n = IOHIDServiceClientCopyProperty(s, CFSTR("Product"));
      if (n) CFStringGetCString(n, name, sizeof name, kCFStringEncodingUTF8);
      if (!strstr(name, "NAND")) continue;
      IOHIDEventRef e = IOHIDServiceClientCopyEvent(s, kIOHIDEventTypeTemperature, 0, 0);
      if (e) printf(" %s=%.1f", name, IOHIDEventGetFloatValue(e, kIOHIDEventTypeTemperature << 16));
    }
    printf("\n"); fflush(stdout);
    if (period <= 0) break;
    sleep(period);
  }
}
