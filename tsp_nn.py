"""Logika algoritma Nearest Neighbor, Dynamic Programming, & rute manual untuk TSP."""
import math
import random
import time

# Koordinat simulasi default (km). Mendukung hingga 14 lokasi dengan jarak berjauhan.
KOORDINAT = [
    (0.8, 1.0), (4.2, 1.5), (8.0, 0.9), (12.0, 1.4), (15.2, 2.2),
    (1.2, 4.8), (5.0, 4.5), (9.2, 4.2), (13.5, 5.0),
    (1.0, 8.6), (4.8, 8.2), (8.6, 8.8), (12.4, 8.0), (15.0, 10.2),
]


def buat_koordinat_tersebar(n, seed, lebar=16.0, tinggi=11.0, min_jarak=2.6):
    """Menghasilkan n koordinat acak berdasarkan seed dengan batas jarak minimum antar node."""
    rng = random.Random(seed)
    titik = []
    jarak_target = min_jarak
    while len(titik) < n:
        berhasil = False
        for _ in range(500):
            x = round(rng.uniform(0.6, lebar - 0.6), 1)
            y = round(rng.uniform(0.6, tinggi - 0.6), 1)
            if all(math.dist((x, y), p) >= jarak_target for p in titik):
                titik.append((x, y))
                berhasil = True
                break
        if not berhasil:
            jarak_target *= 0.9
    return titik


def buat_data(n, koordinat=None):
    """Membuat label, koordinat, dan matriks jarak Euclidean (dibulatkan 1 desimal)."""
    label = [f"LOC{i + 1}" for i in range(n)]
    koord = koordinat[:n] if koordinat is not None else KOORDINAT[:n]
    jarak = [[round(math.dist(a, b), 1) for b in koord] for a in koord]
    return label, koord, jarak


def nearest_neighbor(jarak, label, awal=0):
    """Menjalankan Nearest Neighbor, mengukur runtime (ms), dan mencatat SETIAP langkah."""
    t_mulai = time.perf_counter()

    n = len(jarak)
    rute = [awal]
    belum = [i for i in range(n) if i != awal]
    total = 0.0
    langkah = []

    while belum:
        sekarang = rute[-1]
        kandidat = {j: jarak[sekarang][j] for j in belum}
        tujuan = min(kandidat, key=kandidat.get)
        d = kandidat[tujuan]
        total = round(total + d, 1)
        daftar_belum = list(belum)
        belum.remove(tujuan)
        rute.append(tujuan)
        lain = ", ".join(
            f"{label[j]} ({kandidat[j]:.1f} km)" for j in kandidat if j != tujuan
        )
        alasan = f"{label[tujuan]} dipilih karena jaraknya paling kecil ({d:.1f} km)."
        if lain:
            alasan += f" Kandidat lain lebih jauh: {lain}."
        langkah.append({
            "dari": sekarang,
            "belum": daftar_belum,
            "kandidat": kandidat,
            "tujuan": tujuan,
            "jarak": d,
            "total": total,
            "rute": list(rute),
            "alasan": alasan,
        })

    # Kembali ke depot
    sekarang = rute[-1]
    d = jarak[sekarang][awal]
    total = round(total + d, 1)
    rute.append(awal)
    langkah.append({
        "dari": sekarang,
        "belum": [],
        "kandidat": {awal: d},
        "tujuan": awal,
        "jarak": d,
        "total": total,
        "rute": list(rute),
        "alasan": f"Semua lokasi sudah dikunjungi, kembali ke depot {label[awal]} ({d:.1f} km).",
    })

    waktu_ms = (time.perf_counter() - t_mulai) * 1000.0
    return langkah, waktu_ms


def tsp_dp(jarak, label, awal=0):
    """Menjalankan TSP dengan Dynamic Programming (Bitmask + Memoisasi) & mengukur runtime (ms)."""
    t_mulai = time.perf_counter()

    n = len(jarak)
    full_mask = (1 << n) - 1
    dp = [[None] * (1 << n) for _ in range(n)]
    parent = [[-1] * (1 << n) for _ in range(n)]

    def solve(pos, mask):
        if mask == full_mask:
            return jarak[pos][awal]
        if dp[pos][mask] is not None:
            return dp[pos][mask]

        min_dist = float("inf")
        best_city = -1
        for city in range(n):
            if (mask & (1 << city)) == 0:
                new_dist = round(jarak[pos][city] + solve(city, mask | (1 << city)), 1)
                if new_dist < min_dist:
                    min_dist = new_dist
                    best_city = city

        dp[pos][mask] = min_dist
        parent[pos][mask] = best_city
        return min_dist

    mask_awal = 1 << awal
    solve(awal, mask_awal)

    # Rekonstruksi rute optimal dari matriks parent
    rute = [awal]
    belum = [i for i in range(n) if i != awal]
    total = 0.0
    langkah = []
    pos = awal
    mask = mask_awal

    while belum:
        tujuan = parent[pos][mask]
        kandidat = {j: jarak[pos][j] for j in belum}
        d = jarak[pos][tujuan]
        total = round(total + d, 1)
        daftar_belum = list(belum)

        next_mask = mask | (1 << tujuan)
        sisa_optimal = (
            jarak[tujuan][awal]
            if next_mask == full_mask
            else dp[tujuan][next_mask]
        )
        total_submasalah = round(d + sisa_optimal, 1)

        belum.remove(tujuan)
        rute.append(tujuan)
        alasan = (
            f"{label[tujuan]} dipilih karena menghasilkan total sisa rute paling minimum "
            f"({d:.1f} km + sisa optimal {sisa_optimal:.1f} km = {total_submasalah:.1f} km)."
        )

        langkah.append({
            "dari": pos,
            "belum": daftar_belum,
            "kandidat": kandidat,
            "tujuan": tujuan,
            "jarak": d,
            "total": total,
            "rute": list(rute),
            "alasan": alasan,
        })

        mask = next_mask
        pos = tujuan

    # Kembali ke depot
    d = jarak[pos][awal]
    total = round(total + d, 1)
    rute.append(awal)
    langkah.append({
        "dari": pos,
        "belum": [],
        "kandidat": {awal: d},
        "tujuan": awal,
        "jarak": d,
        "total": total,
        "rute": list(rute),
        "alasan": f"Semua lokasi sudah dikunjungi, kembali ke depot {label[awal]} ({d:.1f} km).",
    })

    waktu_ms = (time.perf_counter() - t_mulai) * 1000.0
    return langkah, waktu_ms


def bangun_langkah_manual(rute_manual, jarak, label, awal):
    """Membangun daftar langkah dari urutan node yang dipilih pengguna secara manual."""
    steps = []
    total = 0.0
    n = len(label)
    for idx in range(len(rute_manual) - 1):
        dari = rute_manual[idx]
        tujuan = rute_manual[idx + 1]
        sudah = set(rute_manual[: idx + 1])
        belum = [j for j in range(n) if j not in sudah] or [awal]
        kandidat = {j: jarak[dari][j] for j in belum}
        d = jarak[dari][tujuan]
        total = round(total + d, 1)
        steps.append({
            "dari": dari,
            "tujuan": tujuan,
            "jarak": d,
            "total": total,
            "rute": rute_manual[: idx + 2],
            "belum": belum if tujuan != awal else [],
            "kandidat": kandidat,
            "alasan": (
                f"Dipilih secara manual dari {label[dari]} menuju {label[tujuan]} "
                f"dengan jarak {d:.1f} km."
            ),
        })
    return steps
