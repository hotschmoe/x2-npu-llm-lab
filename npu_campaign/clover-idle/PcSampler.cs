// PC + frame-pointer-chain sampler for one thread of another process (Windows ARM64).
// Suspends the thread, reads the user-mode CONTEXT (Pc, Lr, Fp), walks the AArch64 frame-record chain
// ({prev_fp, lr} at [fp]) and resumes it. Kernel time shows up as a user PC parked in ntdll!Nt*.
// Compiled by pcsamp.ps1 with Add-Type (C# 5: no string interpolation, no ?. operator).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

public static class PcSampler {
    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr OpenThread(uint access, bool inherit, uint tid);
    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll", SetLastError = true)] static extern uint SuspendThread(IntPtr h);
    [DllImport("kernel32.dll", SetLastError = true)] static extern uint ResumeThread(IntPtr h);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool GetThreadContext(IntPtr h, IntPtr ctx);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool CloseHandle(IntPtr h);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool ReadProcessMemory(IntPtr hp, IntPtr addr, byte[] buf, IntPtr size, out IntPtr read);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool QueryThreadCycleTime(IntPtr h, out ulong cycles);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool GetThreadTimes(IntPtr h, out long c, out long e, out long k, out long u);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct SYMBOL_INFO {
        public uint SizeOfStruct; public uint TypeIndex; public ulong Reserved0; public ulong Reserved1;
        public uint Index; public uint Size; public ulong ModBase; public uint Flags; public ulong Value; public ulong Address;
        public uint Register; public uint Scope; public uint Tag; public uint NameLen; public uint MaxNameLen;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)] public string Name;
    }
    [DllImport("dbghelp.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern bool SymInitializeW(IntPtr hp, string path, bool invade);
    [DllImport("dbghelp.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern bool SymFromAddrW(IntPtr hp, ulong addr, out ulong disp, ref SYMBOL_INFO si);
    [DllImport("dbghelp.dll", SetLastError = true)] static extern bool SymCleanup(IntPtr hp);
    [DllImport("dbghelp.dll", SetLastError = true)] static extern uint SymSetOptions(uint o);

    const uint THREAD_ACCESS = 0x0002 | 0x0008 | 0x0040; // SUSPEND_RESUME | GET_CONTEXT | QUERY_INFORMATION
    const uint PROC_ACCESS = 0x0010 | 0x0400;            // VM_READ | QUERY_INFORMATION
    const int CTX_SIZE = 0x390;
    const uint CONTEXT_ARM64_CONTROL = 0x00400001;

    public static string Info = "";
    public static long KernelTicks, UserTicks; public static ulong Cycles; public static double WallSec;

    // returns list of stacks: [pc, lr, fp-chain...]
    public static List<ulong[]> Sample(int pid, int tid, int durationMs, double intervalMs, int depth) {
        var res = new List<ulong[]>();
        IntPtr ht = OpenThread(THREAD_ACCESS, false, (uint) tid);
        IntPtr hp = OpenProcess(PROC_ACCESS, false, pid);
        if (ht == IntPtr.Zero || hp == IntPtr.Zero) { Info = "open failed " + Marshal.GetLastWin32Error(); return res; }
        IntPtr raw = Marshal.AllocHGlobal(CTX_SIZE + 32);
        IntPtr ctx = new IntPtr((raw.ToInt64() + 15) & ~15L);
        byte[] rec = new byte[16];
        long c0, e0, k0, u0, k1, u1; ulong cy0, cy1;
        GetThreadTimes(ht, out c0, out e0, out k0, out u0); QueryThreadCycleTime(ht, out cy0);
        var sw = Stopwatch.StartNew();
        double next = 0; int fails = 0;
        while (sw.Elapsed.TotalMilliseconds < durationMs) {
            while (sw.Elapsed.TotalMilliseconds < next) { }
            next += intervalMs;
            if (SuspendThread(ht) == 0xFFFFFFFF) { fails++; if (fails > 50) break; continue; }
            Marshal.WriteInt32(ctx, 0, unchecked((int) CONTEXT_ARM64_CONTROL));
            bool ok = GetThreadContext(ht, ctx);
            ulong pc = 0, lr = 0, fp = 0;
            if (ok) {
                fp = (ulong) Marshal.ReadInt64(ctx, 0xF0);
                lr = (ulong) Marshal.ReadInt64(ctx, 0xF8);
                pc = (ulong) Marshal.ReadInt64(ctx, 0x108);
            }
            var st = new List<ulong>(depth + 2);
            st.Add(pc); st.Add(lr);
            for (int d = 0; d < depth && fp != 0 && ok; d++) {
                IntPtr nr;
                if (!ReadProcessMemory(hp, new IntPtr((long) fp), rec, new IntPtr(16), out nr)) break;
                ulong pfp = BitConverter.ToUInt64(rec, 0), plr = BitConverter.ToUInt64(rec, 8);
                st.Add(plr);
                if (pfp <= fp) break;
                fp = pfp;
            }
            ResumeThread(ht);
            if (ok) res.Add(st.ToArray());
        }
        WallSec = sw.Elapsed.TotalSeconds;
        GetThreadTimes(ht, out c0, out e0, out k1, out u1); QueryThreadCycleTime(ht, out cy1);
        KernelTicks = k1 - k0; UserTicks = u1 - u0; Cycles = cy1 - cy0;
        Marshal.FreeHGlobal(raw);
        CloseHandle(ht); CloseHandle(hp);
        Info = "samples=" + res.Count + " fails=" + fails;
        return res;
    }

    // map addresses to "module!symbol+0xoff" using dbghelp (exports / PDBs found on symPath). Process must be alive.
    public static Dictionary<ulong, string> Symbolize(int pid, IEnumerable<ulong> addrs, string symPath) {
        var map = new Dictionary<ulong, string>();
        IntPtr hp = OpenProcess(PROC_ACCESS, false, pid);
        SymSetOptions(0x00000002 | 0x00000004 | 0x00000010); // UNDNAME | DEFERRED_LOADS | LOAD_LINES
        if (!SymInitializeW(hp, symPath, true)) { Info = "SymInitialize failed " + Marshal.GetLastWin32Error(); return map; }
        foreach (var a in addrs) {
            if (map.ContainsKey(a)) continue;
            var si = new SYMBOL_INFO();
            si.SizeOfStruct = 88; // sizeof(SYMBOL_INFOW) without the name buffer (+ 1 WCHAR, aligned)
            si.MaxNameLen = 500;
            ulong disp;
            if (SymFromAddrW(hp, a, out disp, ref si)) map[a] = si.Name + "+0x" + disp.ToString("x");
        }
        SymCleanup(hp); CloseHandle(hp);
        return map;
    }
}
