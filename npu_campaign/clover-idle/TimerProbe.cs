// Measures actual sleep durations of Win32 wait primitives on this machine (no NPU involved).
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

public static class TimerProbe {
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern IntPtr CreateWaitableTimerExW(IntPtr a, string name, uint flags, uint access);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetWaitableTimer(IntPtr h, ref long due, int period, IntPtr cb, IntPtr arg, bool resume);
    [DllImport("kernel32.dll", SetLastError = true)] static extern uint WaitForSingleObject(IntPtr h, uint ms);
    [DllImport("kernel32.dll")] static extern void Sleep(uint ms);
    [DllImport("kernel32.dll")] static extern bool SwitchToThread();

    // mode 0: Sleep(0) ... returns (min, median, p90, max) in microseconds over n iterations
    public static double[] Probe(int mode, long reqUs, int n) {
        IntPtr t = IntPtr.Zero;
        if (mode == 1) t = CreateWaitableTimerExW(IntPtr.Zero, null, 0x2, 0x1F0003);
        var d = new double[n];
        var sw = new Stopwatch();
        for (int i = 0; i < n; i++) {
            sw.Restart();
            if (mode == 0) Sleep((uint) (reqUs / 1000));
            else if (mode == 1) { long due = -reqUs * 10; SetWaitableTimer(t, ref due, 0, IntPtr.Zero, IntPtr.Zero, false); WaitForSingleObject(t, 0xFFFFFFFF); }
            else if (mode == 2) SwitchToThread();
            d[i] = sw.Elapsed.TotalMilliseconds * 1000.0;
        }
        Array.Sort(d);
        return new double[] { d[0], d[n / 2], d[(int) (n * 0.9)], d[n - 1] };
    }
}
