// bandmap: carve a GPU's free VRAM into separate physical chunks (cuMemCreate), pattern-test each,
// report which chunks (and which 64 KiB pages inside them) flip bits, and optionally keep holding
// the bad chunks so no other process can be given that physical memory.
//
// usage: bandmap <dev> <keep_free_mb> <passes> <hold_ms> <mode> [margin_chunks] [chunk_mb]
//   mode test : allocate, test, print, free everything, exit
//   mode hold : allocate, test, free good chunks, keep bad chunks (+/- margin by allocation index),
//               print what is held, then block until SIGINT/SIGTERM
//   mode span : like hold, but keep every chunk from the first bad one to the last (+/- margin)
// Patterns per pass: random (seeded per pass), ones, random-inverted. Zeros is skipped (always clean).
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <csignal>
#include <unistd.h>
#include <chrono>
#include <thread>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(2); } } while (0)

__device__ __forceinline__ uint32_t pat(uint64_t i, uint32_t seed, int mode) {
    if (mode == 1) return 0xffffffffu;
    uint64_t x = i * 0x9E3779B97F4A7C15ull + seed; x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
    return mode == 2 ? ~(uint32_t) x : (uint32_t) x;
}
__global__ void fill(uint32_t* p, uint64_t n, uint32_t seed, int mode) {
    for (uint64_t i = blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; i < n; i += (uint64_t) gridDim.x * blockDim.x)
        p[i] = pat(i, seed, mode);
}
// per 64 KiB page: bad byte count; per byte lane totals; set/cleared bits
__global__ void check(const uint32_t* p, uint64_t n, uint32_t seed, int mode, unsigned int* page_bad,
                      unsigned long long* tot) {
    for (uint64_t i = blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; i < n; i += (uint64_t) gridDim.x * blockDim.x) {
        const uint32_t e = pat(i, seed, mode), v = p[i];
        if (v != e) {
            const uint32_t x = v ^ e;
            for (int b = 0; b < 4; ++b) if ((x >> (8 * b)) & 0xff) atomicAdd(&tot[b], 1ull);
            atomicAdd(&tot[4], (unsigned long long) __popc(x & v));
            atomicAdd(&tot[5], (unsigned long long) __popc(x & e));
            atomicAdd(&page_bad[(i * 4) >> 16], 1u);
        }
    }
}

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s dev keep_free_mb passes hold_ms test|hold|span [margin_chunks] [chunk_mb]\n", argv[0]); return 1; }
    const int dev = atoi(argv[1]);
    const size_t keep = (size_t) atoll(argv[2]) << 20;
    const int passes = atoi(argv[3]);
    const int hold_ms = atoi(argv[4]);
    const bool span = !strcmp(argv[5], "span");
    const bool hold = span || !strcmp(argv[5], "hold");
    const int margin = argc > 6 ? atoi(argv[6]) : 0;
    const size_t want_chunk = argc > 7 ? (size_t) atoll(argv[7]) << 20 : 2ull << 20;

    CK(cuInit(0));
    CUdevice d; CK(cuDeviceGet(&d, dev));
    CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, d)); CK(cuCtxSetCurrent(ctx));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    size_t gmin = 0, grec = 0;
    CK(cuMemGetAllocationGranularity(&gmin, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CK(cuMemGetAllocationGranularity(&grec, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    const size_t chunk = (want_chunk + gmin - 1) / gmin * gmin;

    size_t fr, tot; CK(cuMemGetInfo(&fr, &tot));
    if (fr <= keep + 64 * chunk) { printf("not enough free: %zu MB\n", fr >> 20); return 1; }
    const size_t nmax = (fr - keep) / chunk;
    printf("CUDA%d granularity min %zu KiB rec %zu KiB, chunk %zu KiB; free %zu MB of %zu MB; keeping %zu MB free; up to %zu chunks\n",
           dev, gmin >> 10, grec >> 10, chunk >> 10, fr >> 20, tot >> 20, keep >> 20, nmax);

    CUdeviceptr va; CK(cuMemAddressReserve(&va, nmax * chunk, 0, 0, 0));
    std::vector<CUmemGenericAllocationHandle> h;
    for (size_t i = 0; i < nmax; ++i) {
        CUmemGenericAllocationHandle x;
        if (cuMemCreate(&x, chunk, &prop, 0) != CUDA_SUCCESS) break;
        CK(cuMemMap(va + i * chunk, chunk, 0, x, 0));
        h.push_back(x);
    }
    const size_t n = h.size();
    CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CK(cuMemSetAccess(va, n * chunk, &acc, 1));
    CK(cuMemGetInfo(&fr, &tot));
    printf("mapped %zu chunks = %zu MB; free now %zu MB\n", n, (n * chunk) >> 20, fr >> 20);

    const size_t words = n * chunk / 4, pages = n * chunk >> 16;
    unsigned int* dpage; unsigned long long* dtot;
    CK(cuMemAlloc((CUdeviceptr*) &dpage, pages * 4));
    CK(cuMemAlloc((CUdeviceptr*) &dtot, 8 * 8));
    std::vector<unsigned long long> page_sum(pages, 0);
    std::vector<unsigned int> hp(pages);
    uint32_t* p = (uint32_t*) va;
    const auto t0 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < passes; ++pass) {
        for (int mode = 0; mode < 3; ++mode) {
            const uint32_t seed = 0x51ED + pass * 7919 + mode * 104729;
            fill<<<4096, 256>>>(p, words, seed, mode);
            CK(cuCtxSynchronize());
            std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
            CK(cuMemsetD8((CUdeviceptr) dpage, 0, pages * 4));
            CK(cuMemsetD8((CUdeviceptr) dtot, 0, 64));
            check<<<4096, 256>>>(p, words, seed, mode, dpage, dtot);
            CK(cuCtxSynchronize());
            unsigned long long t[8];
            CK(cuMemcpyDtoH(t, (CUdeviceptr) dtot, 64));
            CK(cuMemcpyDtoH(hp.data(), (CUdeviceptr) dpage, pages * 4));
            size_t badpages = 0;
            for (size_t k = 0; k < pages; ++k) { page_sum[k] += hp[k]; badpages += hp[k] != 0; }
            printf("pass %d %-7s bad bytes per lane [%llu %llu %llu %llu] set %llu cleared %llu, 64KiB pages hit %zu\n",
                   pass, mode == 0 ? "random" : mode == 1 ? "ones" : "~random", t[0], t[1], t[2], t[3], t[4], t[5], badpages);
            fflush(stdout);
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // per-chunk summary
    const size_t ppc = chunk >> 16;
    std::vector<char> bad(n, 0);
    size_t nbad = 0, first = n, last = 0;
    for (size_t c = 0; c < n; ++c) {
        unsigned long long s = 0; size_t hit = 0;
        for (size_t k = 0; k < ppc; ++k) { s += page_sum[c * ppc + k]; hit += page_sum[c * ppc + k] != 0; }
        if (s) {
            bad[c] = 1; ++nbad; if (c < first) first = c; last = c;
            printf("BAD chunk %zu (+%zu MB) bad bytes %llu, 64KiB pages hit %zu/%zu:", c, (c * chunk) >> 20, s, hit, ppc);
            for (size_t k = 0; k < ppc; ++k) if (page_sum[c * ppc + k]) printf(" %zu", k);
            printf("\n");
        }
    }
    printf("SUMMARY CUDA%d tested %zu MB in %zu chunks for %.1f s: %zu bad chunks", dev, (n * chunk) >> 20, n, secs, nbad);
    if (nbad) printf(", allocation index %zu..%zu (+%zu..+%zu MB, span %zu MB)", first, last, (first * chunk) >> 20,
                     (last * chunk) >> 20, ((last - first + 1) * chunk) >> 20);
    printf("\n");
    fflush(stdout);

    CK(cuMemFree((CUdeviceptr) dpage)); CK(cuMemFree((CUdeviceptr) dtot));
    std::vector<char> keepc(n, 0);
    if (span && nbad) for (long j = (long) first - margin; j <= (long) last + margin; ++j) { if (j >= 0 && j < (long) n) keepc[j] = 1; }
    else if (hold) for (size_t c = 0; c < n; ++c) if (bad[c])
        for (long j = (long) c - margin; j <= (long) c + margin; ++j) if (j >= 0 && j < (long) n) keepc[j] = 1;
    CK(cuMemUnmap(va, n * chunk));
    size_t kept = 0;
    for (size_t c = 0; c < n; ++c) {
        if (keepc[c]) { ++kept; continue; }
        CK(cuMemRelease(h[c]));
    }
    CK(cuMemAddressFree(va, nmax * chunk));
    CK(cuMemGetInfo(&fr, &tot));
    printf("%s: holding %zu chunks = %zu MB (margin %d); free now %zu MB\n", hold ? "HOLD" : "DONE", kept,
           (kept * chunk) >> 20, margin, fr >> 20);
    fflush(stdout);
    if (!hold) return 0;
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    while (!g_stop) pause();
    for (size_t c = 0; c < n; ++c) if (keepc[c]) cuMemRelease(h[c]);
    printf("released\n");
    return 0;
}
