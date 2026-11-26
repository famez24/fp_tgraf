"""Menggambar graf dengan NetworkX + Matplotlib bergaya bersih dan proporsional."""
import math
import matplotlib.pyplot as plt
import networkx as nx

LATAR = "#FFFFFF"
GRID_COLOR = "#E2E8F0"
EDGE_BG = "#CBD5E1"
EDGE_KANDIDAT = "#2563EB"
EDGE_RUTE = "#F97316"


def _dalam_lingkaran_luar(p, a, b, c):
    """Mengecek apakah titik p berada di dalam lingkaran luar segitiga (a, b, c)."""
    ax, ay = a[0] - p[0], a[1] - p[1]
    bx, by = b[0] - p[0], b[1] - p[1]
    cx, cy = c[0] - p[0], c[1] - p[1]
    det = (
        (ax * ax + ay * ay) * (bx * cy - cx * by)
        - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
    )
    orient = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return det > 0 if orient > 0 else det < 0


def _pilih_edge_latar(koord):
    """Memilih sisi latar non-silang (triangulasi Delaunay) agar graf rapi seperti diagram."""
    n = len(koord)
    if n <= 3:
        return {(i, j) for i in range(n) for j in range(i + 1, n)}

    xs = [p[0] for p in koord]
    ys = [p[1] for p in koord]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1.0) * 10.0
    mid_x, mid_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

    pts = list(koord) + [
        (mid_x - 2 * span, mid_y - span),
        (mid_x, mid_y + 2 * span),
        (mid_x + 2 * span, mid_y - span),
    ]
    segitiga = [(n, n + 1, n + 2)]

    for idx in range(n):
        p = pts[idx]
        buruk = [t for t in segitiga if _dalam_lingkaran_luar(p, pts[t[0]], pts[t[1]], pts[t[2]])]
        poligon = []
        for t in buruk:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                e_rev = (e[1], e[0])
                if not any(
                    t2 is not t and (e in ((t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])) or e_rev in ((t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])))
                    for t2 in buruk
                ):
                    poligon.append(e)
        segitiga = [t for t in segitiga if t not in buruk]
        for e in poligon:
            segitiga.append((e[0], e[1], idx))

    edges = set()
    for a, b, c in segitiga:
        for u, v in ((a, b), (b, c), (c, a)):
            if u < n and v < n:
                edges.add((min(u, v), max(u, v)))

    if not edges:
        for i in range(n):
            urut = sorted((math.dist(koord[i], koord[j]), j) for j in range(n) if j != i)
            for _, j in urut[:3]:
                edges.add((min(i, j), max(i, j)))
    return edges


def gambar_graf(label, koord, jarak, rute, kandidat=None):
    """rute: urutan indeks yang sudah ditempuh. kandidat: indeks tujuan yang sedang dibandingkan."""
    n = len(label)
    pos = {i: koord[i] for i in range(n)}
    sekarang = rute[-1] if rute else 0
    dikunjungi = set(rute)
    set_kandidat_node = set(kandidat.keys()) if kandidat else set()
    selesai = len(rute) > n

    garis_rute = list(zip(rute[:-1], rute[1:]))
    set_rute = {tuple(sorted(e)) for e in garis_rute}
    edge_kandidat = [(sekarang, k) for k in set_kandidat_node]
    set_kandidat = {tuple(sorted(e)) for e in edge_kandidat}

    # Gabungkan sisi latar bersih (Delaunay) dengan rute aktif & kandidat
    edge_latar = _pilih_edge_latar(koord) - set_rute - set_kandidat
    semua_edge_tampil = sorted(edge_latar | set_rute | set_kandidat)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(semua_edge_tampil)

    padat = n > 10
    ukuran_node = 620 if padat else 740
    font_node = 7.5 if padat else 8.2
    font_edge = 6.8 if padat else 7.4

    # Ukuran kanvas ngepas (proporsional 16:10, tidak kebesaran)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.patch.set_facecolor(LATAR)
    ax.set_facecolor(LATAR)

    # 1. Grid latar tipis seperti kanvas diagram
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID_COLOR, linestyle="-", linewidth=0.75, alpha=0.85)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 2. Sisi latar (abu-abu bersih)
    if edge_latar:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=sorted(edge_latar),
            edge_color=EDGE_BG,
            width=1.25,
            alpha=0.9,
        )

    # 3. Sisi kandidat yang sedang dibandingkan
    if edge_kandidat:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=edge_kandidat,
            edge_color=EDGE_KANDIDAT,
            style="dashed",
            width=1.8,
            alpha=0.95,
        )

    # 4. Sisi rute yang sudah terbentuk
    if garis_rute:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=garis_rute,
            edge_color=EDGE_RUTE,
            width=2.8,
            alpha=1.0,
        )

    # 5. Gambar node bergaya lingkaran bersih ber-outline tegas
    node_biasa, node_kandidat_list, node_hijau, node_aktif = [], [], [], []
    for i in range(n):
        if i == sekarang and not selesai:
            node_aktif.append(i)
        elif i in dikunjungi:
            node_hijau.append(i)
        elif i in set_kandidat_node:
            node_kandidat_list.append(i)
        else:
            node_biasa.append(i)

    if node_biasa:
        nc = nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=node_biasa,
            node_color="#FFFFFF", edgecolors="#0F172A", linewidths=1.6,
            node_size=ukuran_node,
        )
        nc.set_zorder(5)

    if node_kandidat_list:
        nc = nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=node_kandidat_list,
            node_color="#FFFFFF", edgecolors="#1D4ED8", linewidths=2.3,
            node_size=ukuran_node,
        )
        nc.set_zorder(5)

    if node_hijau:
        nc = nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=node_hijau,
            node_color="#DCFCE7", edgecolors="#16A34A", linewidths=1.8,
            node_size=ukuran_node,
        )
        nc.set_zorder(5)

    if node_aktif:
        halo = nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=node_aktif,
            node_color="none", edgecolors="#2563EB", linewidths=1.5,
            node_size=int(ukuran_node * 1.42),
        )
        halo.set_linestyle("--")
        halo.set_zorder(5)

        nc = nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=node_aktif,
            node_color="#FEF3C7", edgecolors="#2563EB", linewidths=2.0,
            node_size=ukuran_node,
        )
        nc.set_zorder(6)

    for i in range(n):
        warna_teks = "#15803D" if i in node_hijau else "#0F172A"
        lbl = nx.draw_networkx_labels(
            G, pos, ax=ax, labels={i: label[i]},
            font_size=font_node, font_color=warna_teks, font_weight="bold",
        )
        for text in lbl.values():
            text.set_zorder(7)

    # 6. Pill label jarak bersih di tengah tiap sisi yang tampil
    for u, v in semua_edge_tampil:
        edge_key = (u, v)
        is_rute = edge_key in set_rute
        is_kandidat = edge_key in set_kandidat

        if is_rute:
            fc, ec, tc, z_ord = "#FFF7ED", "#F97316", "#9A3412", 4
        elif is_kandidat:
            fc, ec, tc, z_ord = "#EFF6FF", "#2563EB", "#1E3A8A", 4
        else:
            fc, ec, tc, z_ord = "#FFFFFF", "#CBD5E1", "#475569", 3

        teks_jarak = nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels={(u, v): f"{jarak[u][v]:.1f}"},
            font_size=font_edge,
            font_color=tc,
            font_weight="bold" if (is_rute or is_kandidat) else "normal",
            label_pos=0.5,
            rotate=False,
            bbox=dict(
                boxstyle="round,pad=0.20",
                fc=fc,
                ec=ec,
                alpha=0.98,
                lw=0.9 if (is_rute or is_kandidat) else 0.75,
            ),
        )
        for text in teks_jarak.values():
            text.set_zorder(z_ord)

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.10)
    fig.tight_layout(pad=0.15)
    return fig