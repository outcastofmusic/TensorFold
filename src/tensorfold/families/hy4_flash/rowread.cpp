// Engram rows read with pread on native threads, the GIL released: a prompt chunk's 32,768 rows a table were ~25,000
// Python preads, each taking the GIL beside the thread launching the chunk's kernels.
#include <torch/extension.h>

#include <atomic>
#include <cerrno>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

// Each read i: size[i] bytes at file offset[i] into out[at[i]:], finishing short reads, on `threads` threads.
void gather(int64_t fd, torch::Tensor offset, torch::Tensor size, torch::Tensor at, torch::Tensor out, int64_t threads) {
  TORCH_CHECK(offset.dtype() == torch::kInt64 && size.dtype() == torch::kInt64 && at.dtype() == torch::kInt64);
  TORCH_CHECK(out.dtype() == torch::kUInt8 && out.is_contiguous() && offset.is_contiguous() && size.is_contiguous()
              && at.is_contiguous());
  const int64_t n = offset.numel();
  const int64_t* o = offset.data_ptr<int64_t>();
  const int64_t* s = size.data_ptr<int64_t>();
  const int64_t* a = at.data_ptr<int64_t>();
  uint8_t* base = out.data_ptr<uint8_t>();
  std::atomic<int64_t> next{0};
  std::atomic<int> failed{0};
  {
    pybind11::gil_scoped_release release;
    auto work = [&]() {
      for (;;) {
        const int64_t i = next.fetch_add(1);
        if (i >= n || failed.load()) return;
        int64_t done = 0;
        while (done < s[i]) {
          const ssize_t r = pread(static_cast<int>(fd), base + a[i] + done, s[i] - done, o[i] + done);
          if (r < 0 && errno == EINTR) continue;
          if (r <= 0) { failed.store(1); return; }
          done += r;
        }
      }
    };
    std::vector<std::thread> pool;
    const int64_t k = std::max<int64_t>(1, std::min<int64_t>(threads, n));
    for (int64_t t = 1; t < k; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
  }
  if (failed.load()) throw std::runtime_error("short read of the Engram table: the checkpoint changed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gather", &gather); }
