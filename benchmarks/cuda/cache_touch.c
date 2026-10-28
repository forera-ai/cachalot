/*
 * Follow-up O (docs/CUDA-PIN-FLUSH-RECORD.md, cell o3): what the CPU does to a pinned buffer's cache lines just before it is copied.
 * Compiled on the GPU host by cache_touch.py with the system `cc` (-O2 -shared -fPIC) and called through ctypes. On a machine that is
 * not x86-64 (the Mac rehearsal) every instruction path reports -1 and ct_features() reports 0, so the cell writes `unavailable` rows.
 *
 *   ct_features()                bit 0 clflush, bit 1 clflushopt, bit 2 clwb, bit 3 sse2 (CPUID leaves 1 and 7)
 *   ct_flush(p, n, kind)         kind 0 clflush + mfence, 1 clflushopt + sfence, 2 clwb + sfence, over every line of [p, p + n)
 *   ct_stream_copy(dst, src, n)  non-temporal 16-byte stores (movntdq) of src into dst, then sfence; dst must be 16-byte aligned
 *   ct_copy(dst, src, n)         an ordinary memcpy (the control for ct_stream_copy)
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__)
#include <cpuid.h>
#include <emmintrin.h>
#endif

#define LINE 64

int ct_features(void) {
#if defined(__x86_64__)
    unsigned a, b, c, d;
    int f = 0;
    if (__get_cpuid(1, &a, &b, &c, &d)) {
        if (d & (1u << 19)) f |= 1;
        if (d & (1u << 26)) f |= 8;
    }
    if (__get_cpuid_max(0, 0) >= 7) {
        __cpuid_count(7, 0, a, b, c, d);
        if (b & (1u << 23)) f |= 2;
        if (b & (1u << 24)) f |= 4;
    }
    return f;
#else
    return 0;
#endif
}

int ct_flush(void *p, size_t n, int kind) {
#if defined(__x86_64__)
    char *q = (char *)((uintptr_t)p & ~(uintptr_t)(LINE - 1));
    char *end = (char *)p + n;
    if (kind == 0) {
        for (; q < end; q += LINE) __asm__ volatile("clflush %0" : "+m"(*(volatile char *)q));
        __asm__ volatile("mfence" ::: "memory");
    } else if (kind == 1) {
        for (; q < end; q += LINE) __asm__ volatile("clflushopt %0" : "+m"(*(volatile char *)q));
        __asm__ volatile("sfence" ::: "memory");
    } else if (kind == 2) {
        for (; q < end; q += LINE) __asm__ volatile("clwb %0" : "+m"(*(volatile char *)q));
        __asm__ volatile("sfence" ::: "memory");
    } else {
        return -2;
    }
    return 0;
#else
    (void)p; (void)n; (void)kind;
    return -1;
#endif
}

int ct_stream_copy(void *dst, const void *src, size_t n) {
#if defined(__x86_64__)
    if (((uintptr_t)dst & 15) != 0) return -2;
    size_t body = n & ~(size_t)15;
    __m128i *d = (__m128i *)dst;
    const char *s = (const char *)src;
    for (size_t i = 0; i < body; i += 16) _mm_stream_si128(d++, _mm_loadu_si128((const __m128i *)(s + i)));
    _mm_sfence();
    if (body < n) memcpy((char *)dst + body, s + body, n - body);
    return 0;
#else
    (void)dst; (void)src; (void)n;
    return -1;
#endif
}

void ct_copy(void *dst, const void *src, size_t n) {
    memcpy(dst, src, n);
}
