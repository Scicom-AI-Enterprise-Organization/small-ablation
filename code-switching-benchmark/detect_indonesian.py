#!/usr/bin/env python3
"""Detect Indonesian-language leakage in code-switching benchmark results.

Why this exists
---------------
`benchmark.py` maps fastText's ``__label__id`` (Indonesian) to ``"malay"``
(see ``LABEL_LANG_MAP``). fastText cannot reliably tell Malaysian Malay apart
from Indonesian, so a model that replies in *Indonesian* instead of proper
Malaysian Malay is still scored as a correct ``"malay"`` match. This script
catches that leak with a curated lexicon of words that are *distinctly
Indonesian* and are NOT valid Malaysian Malay words, so a fluent Malay reply
(e.g. one using ``doktor``, ``nombor``, ``boleh``, ``perkhidmatan``) will not
trip them.

Matching is **whole-token** and case-insensitive. This matters:
  - ``uang`` (money, ID) will NOT match inside ``ruang`` (space, MS).
  - ``aja`` (just, ID) will NOT match inside ``ajakan`` (invitation, MS).
  - ``dokter`` (ID) is distinct from ``doktor`` (MS).
Because matching is exact, some affixed Indonesian forms (``diprioritaskan``,
``beraktivitas``) are missed unless added explicitly below. That is a
deliberate, conservative trade-off: it is a lexicon detector, so it favours
precision (no false positives on genuine Malay) over perfect recall. A reply
written entirely in Indonesian usually contains several markers, so per-reply
detection still works well even when individual inflected forms slip through.

Words deliberately EXCLUDED because they are valid Malay (false friends) or
shared/standard in both languages:
  bisa (MS: venom), izin, paham, mau, cuma, kasih (MS: love), pohon (MS: tree/
  plead), butuh-as-root only via the noun forms, pusing, budak, baja, percuma,
  kereta, banci, gampang, kakak, dah, mas, tidak, dengan, sudah, sangat, ...

Usage
-----
    python detect_indonesian.py                 # the two default model dirs
    python detect_indonesian.py results/<model_dir> [results/<model_dir> ...]
    python detect_indonesian.py --min-hits 2 --show 8 results/<model_dir>
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Indonesian-only lexicon (NOT valid Malaysian Malay)
# Each line gives the Malay equivalent so the list stays auditable.
# ---------------------------------------------------------------------------

rejected_words = [
    # --- Abstract nouns: Indonesian "-itas" vs Malay "-iti" -----------------
    # The single most reliable Indonesian/Malay divergence.
    "aktivitas",       # ms: aktiviti
    "kreativitas",     # ms: kreativiti
    "produktivitas",   # ms: produktiviti
    "efektivitas",     # ms: keberkesanan / efektiviti
    "sensitivitas",    # ms: sensitiviti
    "objektivitas",    # ms: objektiviti
    "kualitas",        # ms: kualiti
    "berkualitas",     # ms: berkualiti
    "kuantitas",       # ms: kuantiti
    "realitas",        # ms: realiti
    "identitas",       # ms: identiti
    "fasilitas",       # ms: fasiliti / kemudahan
    "kapasitas",       # ms: kapasiti
    "komunitas",       # ms: komuniti
    "universitas",     # ms: universiti
    "mayoritas",       # ms: majoriti
    "minoritas",       # ms: minoriti
    "prioritas",       # ms: keutamaan / prioriti
    "popularitas",     # ms: populariti
    "intensitas",      # ms: intensiti
    "validitas",       # ms: kesahan / validiti
    "fleksibilitas",   # ms: fleksibiliti
    "kompleksitas",    # ms: kompleksiti
    "stabilitas",      # ms: kestabilan / stabiliti
    "probabilitas",    # ms: kebarangkalian
    "rutinitas",       # ms: rutin
    "integritas",      # ms: integriti
    "solidaritas",     # ms: perpaduan / solidariti
    "mobilitas",       # ms: mobiliti
    "kapabilitas",     # ms: keupayaan / kapabiliti
    "mentalitas",      # ms: mentaliti
    "formalitas",      # ms: formaliti
    "loyalitas",       # ms: kesetiaan / loyaliti
    "vitalitas",       # ms: vitaliti
    "legalitas",       # ms: kesahihan / legaliti
    "moralitas",       # ms: moraliti
    "kontinuitas",     # ms: kesinambungan
    "kompatibilitas",  # ms: keserasian
    "reliabilitas",    # ms: kebolehpercayaan
    "akuntabilitas",   # ms: akauntabiliti

    # --- Dutch / loanword spelling that Malay renders differently -----------
    "televisi",        # ms: televisyen
    "stasiun",         # ms: stesen
    "bioskop",         # ms: pawagam / panggung wayang
    "provinsi",        # ms: wilayah / negeri
    "konfirmasi",      # ms: pengesahan / sahkan
    "konsumen",        # ms: pengguna
    "rekening",        # ms: akaun
    "kuitansi",        # ms: resit
    "kwitansi",        # ms: resit
    "formulir",        # ms: borang
    "kantor",          # ms: pejabat
    "apotek",          # ms: farmasi
    "apotik",          # ms: farmasi
    "dokter",          # ms: doktor
    "ongkos",          # ms: kos / tambang
    "karcis",          # ms: tiket
    "bensin",          # ms: petrol / minyak
    "sopir",           # ms: pemandu
    "ponsel",          # ms: telefon bimbit
    "gratis",          # ms: percuma
    "diskon",          # ms: diskaun

    # --- Everyday vocabulary: different word entirely -----------------------
    "mobil",           # ms: kereta
    "uang",            # ms: wang / duit
    "celana",          # ms: seluar
    "kulkas",          # ms: peti sejuk
    "handuk",          # ms: tuala
    "becak",           # ms: beca
    "macet",           # ms: sesak / tersekat
    "kendaraan",       # ms: kenderaan
    "bandara",         # ms: lapangan terbang
    "tabungan",        # ms: simpanan
    "ketemu",          # ms: jumpa / bertemu

    # --- Spelling / lexical divergence (formal register) --------------------
    "nomor",           # ms: nombor
    "telepon",         # ms: telefon
    "metode",          # ms: kaedah / metod
    "metoda",          # ms: kaedah / metod
    "jadwal",          # ms: jadual
    "resmi",           # ms: rasmi
    "praktik",         # ms: amalan / praktis
    "standar",         # ms: standard / piawai
    "ekspor",          # ms: eksport
    "impor",           # ms: import
    "napas",           # ms: nafas
    "pikir",           # ms: fikir
    "karena",          # ms: kerana
    # NB: silakan / silahkan removed — Malaysian Malay uses them too.
    "berbeda",         # ms: berbeza
    "beda",            # ms: beza
    "perbedaan",       # ms: perbezaan
    "bahwa",           # ms: bahawa
    "kode",            # ms: kod
    "kasus",           # ms: kes
    "otomatis",        # ms: automatik
    "populer",         # ms: popular
    "bisnis",          # ms: bisnes / perniagaan
    "persentase",      # ms: peratus / peratusan
    "kampanye",        # ms: kempen
    "listrik",         # ms: elektrik
    "teknis",          # ms: teknikal
    "pria",            # ms: lelaki
    "rusak",           # ms: rosak
    "kebutuhan",       # ms: keperluan
    "membutuhkan",     # ms: memerlukan
    "mencoba",         # ms: mencuba
    "coba",            # ms: cuba

    # --- Common affixed forms (added explicitly; exact match misses them) ---
    "beraktivitas",    # ms: beraktiviti
    "diprioritaskan",  # ms: diutamakan
    "memprioritaskan", # ms: mengutamakan

    # --- Colloquial / slang: unmistakably Indonesian ------------------------
    "nggak",           # ms: tak / tidak
    "enggak",          # ms: tak / tidak
    "ngga",            # ms: tak / tidak
    "kagak",           # ms: tak / tidak
    "banget",          # ms: sangat / amat
    "aja",             # ms: saja
    "kayak",           # ms: macam / seperti
    "gimana",          # ms: macam mana / bagaimana
    "doang",           # ms: saja
    # NB: bikin removed — Malaysian Malay uses it colloquially too.
    "ngomong",         # ms: cakap / berkata
    "ngomongin",       # ms: bercakap tentang
    "omong",           # ms: cakap
    "ngerti",          # ms: faham
    "ngeliat",         # ms: tengok / lihat
    "ngasih",          # ms: bagi / beri
    "dikasih",         # ms: diberi
    "nyari",           # ms: cari
    "pengen",          # ms: nak / mahu
    "pengin",          # ms: nak / mahu
    "pingin",          # ms: nak / mahu
    "gue",             # ms: saya / aku
    # NB: gua dropped — it means "cave" in Malay (false friend), not just "I".
    "elo",             # ms: awak / kamu
    "elu",             # ms: awak / kamu
    # NB: 2-letter slang (gw, lu, lo, ga, mah) dropped — they collide with
    # units / codes once digits are stripped (mAh, GW, flight code GA).
    "doi",             # ms: dia / kekasih
    "bokap",           # ms: ayah / bapak
    "nyokap",          # ms: ibu / emak
    "tante",           # ms: makcik
    "udah",            # ms: dah / sudah
    "udh",             # ms: dah / sudah
    "belom",           # ms: belum
    "kalo",            # ms: kalau
    "gitu",            # ms: macam itu / begitu
    "gini",            # ms: macam ini / begini
    "emang",           # ms: memang
    "emangnya",        # ms: memangnya
    "bener",           # ms: benar / betul
    "makasih",         # ms: terima kasih
    "mbak",            # ms: kakak / cik
    "ngapain",         # ms: buat apa / kenapa
    "apaan",           # ms: apa
    "ntar",            # ms: nanti
    "entar",           # ms: nanti
    "cuman",           # ms: cuma / hanya
    "besok",           # ms: esok
    "kemarin",         # ms: semalam / kelmarin
    # NB: short discourse particles (sih, dong, deh, nih, tuh, kok, loh, lho,
    # toh) and short negations (gak, kaga) dropped — they collide with
    # romanised Tamil in this corpus, e.g. "iya"=ஐயா (sir), "kaga"=காக (for),
    # "tuh" — causing false positives. Negation is covered by nggak/enggak/ngga.
    "engga",           # ms: tak / tidak
    "gpp",             # ms: tak apa (gak apa-apa)
    "yuk",             # ms: jom
    "yok",             # ms: jom
    "kuy",             # ms: jom (yuk reversed)

    # --- Indonesian everyday / "subtitle" vocabulary (NOT Malaysian) --------
    "cewek",           # ms: perempuan / awek
    "cowok",           # ms: lelaki / pakwe
    "kamar",           # ms: bilik
    "kursi",           # ms: kerusi
    "lemari",          # ms: almari
    "sepatu",          # ms: kasut
    "kaos",            # ms: baju-T / kemeja-T
    "sepeda",          # ms: basikal
    "pulsa",           # ms: kredit / topup
    "jajan",           # ms: beli snek
    "cemilan",         # ms: snek / kudapan
    "gede",            # ms: besar
    "keren",           # ms: hebat / mantap
    "jelek",           # ms: buruk / hodoh
    "cakep",           # ms: kacak / cantik
    "capek",           # ms: penat / letih
    "cape",            # ms: penat / letih
    "ribet",           # ms: rumit / leceh
    "repot",           # ms: menyusahkan
    "betah",           # ms: selesa / suka berada
    "jorok",           # ms: kotor / jijik
    "boong",           # ms: bohong / tipu
    "gemes",           # ms: gemas
    "asik",            # ms: asyik / seronok
    "mending",         # ms: lebih baik
    "mumpung",         # ms: selagi ada peluang
    "mulu",            # ms: asyik / selalu (melulu)
    "cuek",            # ms: tak peduli
    "gengsi",          # ms: maruah / segan
    "jutek",           # ms: ketus / sombong

    # --- Indonesian slang (social media / subtitle register) ---------------
    "baper",           # bawa perasaan -> ms: terasa hati
    "gabut",           # gaji buta -> ms: tak ada kerja / bosan
    "mager",           # malas gerak -> ms: malas bergerak
    "bokek",           # ms: kering / tiada duit
    "gercep",          # gerak cepat -> ms: bertindak pantas
    "gokil",           # ms: gila / hebat
    "mantul",          # mantap betul -> ms: mantap
    "nongkrong",       # ms: lepak
    "nyantai",         # ms: bersantai / relaks
    "jomblo",          # ms: bujang / single
    "gebetan",         # ms: orang yang diminati
    "bete",            # bad temper -> ms: bosan / menyampah
    "sebel",           # ms: meluat / geram
    "kesel",           # ms: geram / jengkel
    "ngambek",         # ms: merajuk
    "anjir",           # ID expletive (anjing softened)
    "anjay",           # ID expletive / interjection

    # --- More "-itas" abstract nouns (the rule below catches the rest) ------
    "totalitas",       # ms: totaliti
    "nasionalitas",    # ms: kewarganegaraan / nasionaliti
    "rasionalitas",    # ms: rasionaliti
    "mortalitas",      # ms: mortaliti / kematian
    "kriminalitas",    # ms: jenayah
    "relativitas",     # ms: relativiti
    "subjektivitas",   # ms: subjektiviti
    "eksklusivitas",   # ms: eksklusiviti
    "kredibilitas",    # ms: kredibiliti
    "aksesibilitas",   # ms: kebolehcapaian / aksesibiliti
    "fungsionalitas",  # ms: kefungsian / fungsionaliti
    "personalitas",    # ms: personaliti
    "legitimitas",     # ms: kesahihan / legitimasi
    "otoritas",        # ms: autoriti / pihak berkuasa
    "seksualitas",     # ms: seksualiti
    "spiritualitas",   # ms: kerohanian / spiritualiti
    "komoditas",       # ms: komoditi
    "obesitas",        # ms: obesiti
    "diversitas",      # ms: kepelbagaian / diversiti

    # --- Systematic loanword / spelling divergences -------------------------
    "obat",            # ms: ubat
    "hewan",           # ms: haiwan
    "tivi",            # ms: tv / televisyen
    "perawat",         # ms: jururawat
    "suster",          # ms: jururawat
    "puskesmas",       # ms: klinik kesihatan
    "pulpen",          # ms: pen
    "ulangan",         # ms: ujian
    "mie",             # ms: mi
    "rok",             # ms: skirt
    "jilbab",          # ms: tudung
    "kerudung",        # ms: tudung
    "kacamata",        # ms: cermin mata
    "kantong",         # ms: kocek / poket
    "kasur",           # ms: tilam
    "sendok",          # ms: sudu
    "panci",           # ms: periuk
    "kompor",          # ms: dapur / tungku
    "colokan",         # ms: soket / palam
    "saklar",          # ms: suis
    "odol",            # ms: ubat gigi
    "ngepel",          # ms: mengemop
    "klakson",         # ms: hon
    "setir",           # ms: stereng
    "angkot",          # ms: bas mini
    "parkir",          # ms: parking / letak kereta
    "sortir",          # ms: asingkan / isih
    "struk",           # ms: resit
    "kembalian",       # ms: baki / duit kembali
    "receh",           # ms: duit syiling
    "keuangan",        # ms: kewangan
    "gajian",          # ms: hari gaji
    "lembur",          # ms: kerja lebih masa
    "obral",           # ms: jualan murah
    "belanjaan",       # ms: barang belian
    "nasabah",         # ms: pelanggan (bank)
    "karyawan",        # ms: pekerja / kakitangan
    "pribumi",         # ms: bumiputera
    "bule",            # ms: orang putih / mat salleh
    "gorengan",        # ms: makanan goreng
    "asin",            # ms: masin
    "pilek",           # ms: selesema

    # --- Days / months that diverge ----------------------------------------
    "senin",           # ms: isnin
    "kamis",           # ms: khamis
    "jumat",           # ms: jumaat

    # --- Jakarta colloquial respellings (never written this way in Malay) ---
    "inget",           # ms: ingat
    "nginget",         # ms: mengingat
    "dateng",          # ms: datang
    "denger",          # ms: dengar
    "dengerin",        # ms: dengarkan
    "seneng",          # ms: senang / gembira
    "deket",           # ms: dekat
    "cepet",           # ms: cepat
    "bentar",          # ms: sebentar
    "dikit",           # ms: sedikit
    "rame",            # ms: ramai
    "pinter",          # ms: pintar
    "laper",           # ms: lapar
    "nunggu",          # ms: tunggu
    "nungguin",        # ms: menunggu
    "trus",            # ms: terus
    # NB: iya dropped — collides with romanised Tamil "iya" (ஐயா, sir).

    # --- Colloquial nge-/ny- verbs ------------------------------------------
    "ngecek",          # ms: semak / check
    "mikir",           # ms: fikir
    "kepengen",        # ms: teringin
    "ngobrol",         # ms: berbual / sembang
    "obrolan",         # ms: perbualan
    "curhat",          # ms: meluah perasaan
    "ngomel",          # ms: bebel
    "nyuruh",          # ms: suruh
    "nginep",          # ms: bermalam
    "mampir",          # ms: singgah
    "ngajak",          # ms: mengajak
    "ngajakin",        # ms: mengajak
    "nganter",         # ms: hantar
    "nganterin",       # ms: menghantar
    "nyampe",          # ms: sampai
    "ngirim",          # ms: kirim / hantar
    "ngirimin",        # ms: mengirimkan
    "ngabarin",        # ms: beritahu / maklumkan
    "nelpon",          # ms: menelefon
    "pesen",           # ms: pesan / tempah
    "setor",           # ms: deposit / masuk duit
    "nyetor",          # ms: mendeposit
    "nabung",          # ms: menyimpan duit
    "kelupaan",        # ms: terlupa
    "nyolong",         # ms: mencuri
    "maling",          # ms: pencuri

    # --- More adjectives / people (distinctly ID) ---------------------------
    "ganteng",         # ms: kacak / handsome
    "imut",            # ms: comel
    "goblok",          # ms: bodoh
    "bocah",           # ms: budak / kanak-kanak
    "preman",          # ms: samseng
    "jagoan",          # ms: jaguh / wira
    "makasi",          # ms: terima kasih
    "permisi",         # ms: tumpang lalu / minta diri
    "makanya",         # ms: sebab itu
    "pokoknya",        # ms: yang penting / pendek kata
    "intinya",         # ms: pada intinya
    "kayaknya",        # ms: nampaknya / agaknya

    # --- High-frequency function / pronoun / discourse ----------------------
    "kalian",          # ms: kamu semua / anda semua
    "bagian",          # ms: bahagian
    "sebagian",        # ms: sebahagian
    "lagian",          # ms: lagipun
    "mendingan",       # ms: lebih baik
    "gausah",          # ms: tak payah
    "yaudah",          # ms: dahlah / sudahlah
    "barusan",         # ms: baru tadi
    "telat",           # ms: lambat / lewat
    "dapet",           # ms: dapat
    "pake",            # ms: pakai

    # --- Banking / insurance / finance --------------------------------------
    "asuransi",        # ms: insurans
    "cicilan",         # ms: ansuran / bayaran ansuran
    "angsuran",        # ms: ansuran
    "kartu",           # ms: kad
    "nomer",           # ms: nombor
    "setoran",         # ms: deposit
    "utang",           # ms: hutang
    "materai",         # ms: setem hasil  (NB: ms "meterai" = seal, excluded)
    "antre",           # ms: beratur / giliran
    "antri",           # ms: beratur / giliran
    "antrean",         # ms: barisan / giliran
    "kasir",           # ms: juruwang

    # --- Tech / devices -----------------------------------------------------
    "fitur",           # ms: ciri / fungsi
    "tombol",          # ms: butang
    "baterai",         # ms: bateri
    "unduh",           # ms: muat turun
    "unggah",          # ms: muat naik
    "unduhan",         # ms: muat turun
    "unggahan",        # ms: muat naik
    "tautan",          # ms: pautan
    "ngecas",          # ms: mengecas

    # --- Medical ------------------------------------------------------------
    "resep",           # ms: preskripsi / ubat

    # --- Education ----------------------------------------------------------
    "dosen",           # ms: pensyarah
    "beasiswa",        # ms: biasiswa
    "rapor",           # ms: kad laporan

    # --- Places / Indonesian admin divisions --------------------------------
    "toko",            # ms: kedai
    "kecamatan",       # ms: daerah / mukim
    "kelurahan",       # ms: mukim
    "kabupaten",       # ms: daerah
    "losmen",          # ms: rumah tumpangan
    "kosan",           # ms: bilik sewa
    "kost",            # ms: bilik sewa
    "halte",           # ms: perhentian bas
    "ojek",            # ms: motosikal sewa
    "begal",           # ms: perompak / penyamun

    # --- Household ----------------------------------------------------------
    "seprai",          # ms: cadar
    "sprei",           # ms: cadar
    "gorden",          # ms: langsir
    "wastafel",        # ms: besen / sinki
    "keran",           # ms: paip
    "bohlam",          # ms: mentol
    "stopkontak",      # ms: soket
    "nyuci",           # ms: mencuci / membasuh

    # --- Food / clothing ----------------------------------------------------
    "bakso",           # ms: bebola daging
    "wortel",          # ms: lobak merah
    "gurih",           # ms: sedap / lemak
    "kaus",            # ms: baju-T
    "dasi",            # ms: tali leher

    # --- Family / people ----------------------------------------------------
    "istri",           # ms: isteri
    "kakek",           # ms: datuk / atuk
    "keponakan",       # ms: anak saudara
    "temen",           # ms: kawan / teman
    "ortu",            # ms: ibu bapa (orang tua)

    # --- Feelings / adjectives ----------------------------------------------
    "kaget",           # ms: terkejut
    "khawatir",        # ms: khuatir / risau
    "kuatir",          # ms: khuatir / risau
    "lemot",           # ms: lembab / perlahan

    # --- Colloquial verbs (ng-/ny-/-in) -------------------------------------
    "ngebut",          # ms: memandu laju
    "nyetir",          # ms: memandu
    "nanya",           # ms: tanya
    "nanyain",         # ms: bertanya
    "bantuin",         # ms: tolong / bantu
    "benerin",         # ms: betulkan / baiki
    "beresin",         # ms: kemaskan / selesaikan
    "ngambil",         # ms: ambil
    "naruh",           # ms: letak
    "bayarin",         # ms: bayarkan
    "mesen",           # ms: pesan / tempah
    "ketemuan",        # ms: berjumpa
    "mudik",           # ms: balik kampung

    # --- Indonesian-only spelling variants ----------------------------------
    "praktek",         # ms: praktik / amalan
    "nasehat",         # ms: nasihat

    # --- Call-centre / customer-service terms -------------------------------
    # Billing & finance
    "biaya",           # ms: kos / caj / yuran
    "iuran",           # ms: yuran / caruman
    "tagihan",         # ms: bil
    "melunasi",        # ms: menjelaskan / melangsaikan
    "pelunasan",       # ms: penyelesaian / pelangsaian
    "pengembalian",    # ms: pemulangan (refund)
    # Telco / connectivity
    "sinyal",          # ms: isyarat / signal
    "koneksi",         # ms: sambungan
    "terhubung",       # ms: tersambung
    "aktivasi",        # ms: pengaktifan
    # Account / policy / terms
    "kebijakan",       # ms: dasar / polisi
    "ketentuan",       # ms: terma / syarat / peruntukan
    "kadaluarsa",      # ms: luput / tamat tempoh
    "kadaluwarsa",     # ms: luput / tamat tempoh
    "kedaluwarsa",     # ms: luput / tamat tempoh
    "berlangganan",    # ms: melanggan / langganan
    "registrasi",      # ms: pendaftaran
    "pemesanan",       # ms: tempahan
    "terkirim",        # ms: dihantar / terhantar
    "mengonfirmasi",   # ms: mengesahkan
    "dikonfirmasi",    # ms: disahkan
    "pengecekan",      # ms: semakan
    # Service flow / complaints / feedback
    "kendala",         # ms: masalah / kekangan / halangan
    "keluhan",         # ms: aduan
    "pengaduan",       # ms: aduan
    "menindaklanjuti", # ms: tindakan susulan / menyusuli
    "rincian",         # ms: butiran / perincian
    "ketidaknyamanan", # ms: ketidakselesaan
    "survei",          # ms: tinjauan / kaji selidik

    # --- Call-centre: insurance / banking / loans ---------------------------
    "pertanggungan",   # ms: perlindungan / liputan
    "mengajukan",      # ms: mengemukakan / membuat permohonan
    "pengajuan",       # ms: permohonan
    "santunan",        # ms: pampasan / khairat
    "tertanggung",     # ms: pihak yang dilindungi
    "agunan",          # ms: cagaran
    "plafon",          # ms: had / siling kredit
    # --- Call-centre: airline / travel --------------------------------------
    "maskapai",        # ms: syarikat penerbangan
    "keterlambatan",   # ms: kelewatan
    "penjadwalan",     # ms: penjadualan
    "rute",            # ms: laluan
    # NB: keberangkatan/penundaan/ditunda removed — borderline with Malay
    # ("keberangkatan tiba" is formal Malay; Malay has "menunda"/"tunda").
    # Flight delay is still covered by the clearly-ID "tertunda".
    # --- Call-centre: telco / connectivity ----------------------------------
    "kecepatan",       # ms: kelajuan
    "operasional",     # ms: operasi (waktu operasi)
    # --- Call-centre: healthcare --------------------------------------------
    "pasien",          # ms: pesakit
    "bpjs",            # ms: (insurans kesihatan Indonesia)
    # --- Call-centre: account / status / process verbs ----------------------
    "terdaftar",       # ms: berdaftar
    "nonaktif",        # ms: tidak aktif
    "menonaktifkan",   # ms: menyahaktifkan / mematikan
    "blokir",          # ms: sekat / blok
    "memblokir",       # ms: menyekat
    "berhasil",        # ms: berjaya
    "sukses",          # ms: berjaya
    "terjadwal",       # ms: dijadualkan
    "pembaruan",       # ms: kemaskini
    "memperbarui",     # ms: mengemaskini
    "diperbarui",      # ms: dikemaskini
    "perbarui",        # ms: kemaskini
    "menginformasikan",# ms: memaklumkan
    "mengabari",       # ms: memberitahu / mengkhabarkan
    "merubah",         # ms: mengubah
    "fotokopi",        # ms: fotostat / salinan
    "stempel",         # ms: cop / setem
    "ongkir",          # ms: kos penghantaran (ongkos kirim)
]

# Multi-word / hyphenated markers (matched with word boundaries) -------------
rejected_phrases = [
    "pertama-tama",    # ms: pertama sekali
    "rumah sakit",     # ms: hospital
    "kebun binatang",  # ms: zoo / taman haiwan
    "kamar mandi",     # ms: bilik air / tandas
    "sampai jumpa",    # ms: jumpa lagi
    "kantor pos",      # ms: pejabat pos
    "mesin cuci",      # ms: mesin basuh
    "sepeda motor",    # ms: motosikal
    "selamat siang",   # ms: selamat tengah hari
    "selamat sore",    # ms: selamat petang
    "rawat inap",      # ms: rawatan dalam wad
    "rawat jalan",     # ms: rawatan pesakit luar
    "makan siang",     # ms: makan tengah hari
    "kaus kaki",       # ms: stoking / sarung kaki
    # Call-centre phrases
    "jatuh tempo",     # ms: tarikh matang / tempoh matang
    "isi ulang",       # ms: tambah nilai / topup
    "tarik tunai",     # ms: pengeluaran tunai
    "syarat dan ketentuan",  # ms: terma dan syarat
    "umpan balik",     # ms: maklum balas
    "tindak lanjut",   # ms: tindakan susulan
    "kata sandi",      # ms: kata laluan
    "layanan pelanggan",     # ms: khidmat / perkhidmatan pelanggan
    "suku bunga",      # ms: kadar faedah / kadar bunga
    "gawat darurat",   # ms: kecemasan
    "masa tenggang",   # ms: tempoh tangguh (grace period)
    "data diri",       # ms: maklumat / butiran peribadi
]

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

_REJECTED = frozenset(w.lower() for w in rejected_words)
_WORD_RE = re.compile(r"[a-zà-ÿ]+")
_PHRASE_RES = [
    (ph, re.compile(r"\b" + re.escape(ph) + r"\b")) for ph in rejected_phrases
]

# Productive suffix rule: Indonesian abstract nouns end in "-itas" where
# Malaysian Malay uses "-iti" (kualitas/kualiti, prioritas/prioriti, ...).
# Malay never writes "-itas", so any "-itas" token is Indonesian. This catches
# the long tail (elastisitas, viskositas, komoditas, ...) without listing each.
# Requires >= 3 letters before "itas" so "tas" (bag), "batas"/"atas" (which end
# in -atas, shared) are NOT matched.
_RULE_RES = [
    ("-itas", re.compile(r"^[a-z]{3,}itas(?:nya)?$")),
]
# Latin/English words ending "-itas" that the rule must NOT flag in English text.
_RULE_EXCLUDE = {"gravitas", "veritas", "caritas", "civitas"}


def extract_reply(text: str) -> str:
    """Return only the assistant's actual reply, dropping any reasoning.

    Reasoning models (e.g. Qwen here) emit an English chain-of-thought wrapped
    in ``<think>...</think>`` before the real answer. We must not scan that
    English planning text for Indonesian markers (it produces false positives
    such as a ``Konfirmasi`` section header). Everything after the last
    ``</think>`` is the reply. Models without the tag (e.g. Gemma) are scanned
    whole.
    """
    marker = "</think>"
    idx = text.rfind(marker)
    return text[idx + len(marker):] if idx != -1 else text


def find_indonesian(text: str) -> Counter:
    """Return a Counter of Indonesian marker -> number of occurrences in text."""
    low = text.lower()
    hits: Counter = Counter()
    for tok in _WORD_RE.findall(low):
        if tok in _REJECTED:
            hits[tok] += 1
            continue
        if tok in _RULE_EXCLUDE:
            continue
        for _, rx in _RULE_RES:
            if rx.match(tok):
                hits[tok] += 1
                break
    for ph, rx in _PHRASE_RES:
        n = len(rx.findall(low))
        if n:
            hits[ph] += n
    return hits


# ---------------------------------------------------------------------------
# Result-file scanning & reporting
# ---------------------------------------------------------------------------


def iter_records(progress_file: Path):
    with open(progress_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def analyse_dir(result_dir: Path, min_hits: int, show: int) -> dict:
    progress_file = result_dir / "progress.jsonl"
    if not progress_file.exists():
        raise FileNotFoundError(f"No progress.jsonl in {result_dir}")

    total = 0
    flagged = 0
    strong = 0                                     # >= 2 distinct markers
    word_counter: Counter = Counter()
    dist_dist: Counter = Counter()                 # distinct markers -> count
    by_switch: dict[str, list[int]] = {}          # lang -> [total, flagged]
    malay_label_total = 0                          # fastText said "malay"
    malay_label_flagged = 0
    match_malay_total = 0                           # scored correct, switch==malay
    match_malay_flagged = 0
    examples: list[dict] = []

    for rec in iter_records(progress_file):
        resp = rec.get("response", "") or ""
        if not resp.strip():
            continue
        total += 1

        reply = extract_reply(resp)
        hits = find_indonesian(reply)
        n_distinct = len(hits)                      # distinct marker words
        dist_dist[min(n_distinct, 3)] += 1
        is_id = n_distinct >= min_hits

        sw = rec.get("switch_language", "unknown")
        by_switch.setdefault(sw, [0, 0])
        by_switch[sw][0] += 1

        if rec.get("response_lang") == "malay":
            malay_label_total += 1
        if rec.get("match") and sw == "malay":
            match_malay_total += 1

        if is_id:
            flagged += 1
            if n_distinct >= 2:
                strong += 1
            word_counter.update(hits)
            by_switch[sw][1] += 1
            if rec.get("response_lang") == "malay":
                malay_label_flagged += 1
            if rec.get("match") and sw == "malay":
                match_malay_flagged += 1
            if len(examples) < show:
                first = next(iter(hits))
                pos = reply.lower().find(first)
                snippet = reply[max(0, pos - 90):pos + 90].replace("\n", " ")
                examples.append({
                    "index": rec.get("index"),
                    "switch_language": sw,
                    "response_lang": rec.get("response_lang"),
                    "n_distinct": n_distinct,
                    "markers": dict(hits),
                    "snippet": snippet.strip(),
                })

    return {
        "dir": result_dir.name,
        "total": total,
        "flagged": flagged,
        "strong": strong,
        "dist_dist": dist_dist,
        "word_counter": word_counter,
        "by_switch": by_switch,
        "malay_label_total": malay_label_total,
        "malay_label_flagged": malay_label_flagged,
        "match_malay_total": match_malay_total,
        "match_malay_flagged": match_malay_flagged,
        "examples": examples,
    }


def _pct(n: int, d: int) -> str:
    return f"{(100 * n / d):.2f}%" if d else "n/a"


def print_report(r: dict, min_hits: int) -> None:
    print("=" * 72)
    print(f"Model dir : {r['dir']}")
    print(f"Responses : {r['total']} (non-empty; reasoning <think> stripped)")
    print(f"Indonesian leak (>= {min_hits} distinct marker word"
          f"{'s' if min_hits > 1 else ''}): "
          f"{r['flagged']}  ({_pct(r['flagged'], r['total'])})")
    print(f"  of which >= 2 distinct markers (likely a full Indonesian reply): "
          f"{r['strong']}  ({_pct(r['strong'], r['total'])})")

    print("\nDistinct-marker distribution (distinct Indonesian words per reply):")
    for k in sorted(r["dist_dist"]):
        label = "3+" if k == 3 else str(k)
        print(f"  {label:>2} distinct: {r['dist_dist'][k]}")

    print("\nBy target (switch) language:")
    for lang, (tot, flg) in sorted(r["by_switch"].items()):
        print(f"  {lang:<10}: {flg:>4}/{tot:<4} Indonesian  ({_pct(flg, tot)})")

    print("\nImpact on the 'malay' score (the words that fastText conflates):")
    print(f"  fastText-labelled 'malay' replies : {r['malay_label_total']}")
    print(f"    of which actually Indonesian    : {r['malay_label_flagged']}  "
          f"({_pct(r['malay_label_flagged'], r['malay_label_total'])})")
    print(f"  malay-target replies scored CORRECT: {r['match_malay_total']}")
    print(f"    but actually Indonesian (= false): {r['match_malay_flagged']}  "
          f"({_pct(r['match_malay_flagged'], r['match_malay_total'])})")

    print("\nTop markers:")
    for w, c in r["word_counter"].most_common(20):
        print(f"  {w:<16} {c}")

    if r["examples"]:
        print("\nExamples:")
        for ex in r["examples"]:
            print(f"  [idx {ex['index']}] target={ex['switch_language']} "
                  f"fasttext={ex['response_lang']} distinct={ex['n_distinct']} "
                  f"markers={ex['markers']}")
            print(f"      ...{ex['snippet']}...")
    print("=" * 72 + "\n")


DEFAULT_DIRS = [
    "results/Qwen_Qwen3.6-27B_0528",
    "results/google_gemma-4-31B-it_0528",
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="*", help="Result dirs (default: the two ablation dirs)")
    p.add_argument("--min-hits", type=int, default=1,
                   help="Min Indonesian marker words to flag a reply (default: 1)")
    p.add_argument("--show", type=int, default=5,
                   help="Number of example replies to print per dir (default: 5)")
    args = p.parse_args()

    dirs = args.dirs or [str(SCRIPT_DIR / d) for d in DEFAULT_DIRS]
    for d in dirs:
        path = Path(d)
        if not path.is_absolute():
            path = SCRIPT_DIR / d
        report = analyse_dir(path, args.min_hits, args.show)
        print_report(report, args.min_hits)


if __name__ == "__main__":
    main()
