# Guideline Normalisasi Task 1

Versi: v2 (2026-11-04)
Untuk label LLM (`llm_types`, `llm_slots`) dan QC human (`review_mln`, `llm_types`, `gold_conda`).

## Esensi

Kata yang sengaja disamarkan dikembalikan ke bentuk asli. Berlaku untuk semua kata, tidak hanya kata kasar.

## Unit

- Satu slot = satu token raw. Tiga huruf tunggal berurutan atau lebih (`f u c k`) = satu slot.
- Output per slot: lowercase, boleh lebih dari satu kata.
- Slot tidak digabung, dipecah, atau dikosongkan.
- Target: bahasa Inggris baku, gaya MultiLexNorm.
- Ragu -> biarkan.

## Diubah

| Jenis | Contoh |
| --- | --- |
| Leet (angka sebagai huruf) | b1tch -> bitch, h0w -> how, n00b -> noob, 5h1t -> shit, a55 -> ass |
| Simbol sebagai huruf | $tupid -> stupid, !diot -> idiot, d@mn -> damn, sh!t -> shit, jacka$$ -> jackass |
| Simbol penyisip | f*ck -> fuck, f.u.c.k -> fuck, b-i-t-c-h -> bitch |
| Leet + huruf berulang | y0o0o -> yo, n0000b -> noob, fuuuck -> fuck |
| Huruf dipisah spasi | f u c k -> fuck, n i k i t a -> nikita, u r a -> you are a |
| Aksen, homoglyph | köll -> kill, b¡tch -> bitch, ßig -> big |
| Huruf hilang | idot -> idiot, stupd -> stupid, fck -> fuck |
| Typo, huruf tertukar | jsut -> just, reprot -> report, gmae -> game, teh -> the |
| Singkatan chat | sry -> sorry, rly -> really, tnx -> thanks, dunno -> don't know, w/e -> whatever |
| Apostrof hilang | shouldnt -> shouldn't, ure -> you're, i`m -> i'm |
| Huruf berulang | nooob -> noob, noooo -> no, plss -> please |
| Leet fonetik | w0w -> wow, gr8 -> great, m8 -> mate, n1 -> nice one |
| Kata kasar salah eja | fking -> fucking, fuckign -> fucking, sht -> shit, tard -> retard |
| Varian istilah game | izi -> ez, gege -> gg, wpo -> wp, supp -> support |
| Nama hero/item salah eja | zues -> zeus, pheonix -> phoenix, meppo -> meepo |
| Satuan | lvl -> level, dmg -> damage, secs -> seconds, 2mins -> 2 minutes |

## Dibiarkan

| Jenis | Contoh |
| --- | --- |
| Istilah game lexicon CONDA | wk, sf, bkb, mid, gg, ez |
| Singkatan hero, role, mekanik | ns, ls, ud, fv, gpm, int, bg, hf |
| Akronim yang tetap di MultiLexNorm | wtf, stfu, idk, omg, lol, ffs, tfw |
| Tawa, interjeksi | jaja, kek, lel, xaxa, wahaha, meh, ouch, shh, pff |
| Emote Twitch | 4head, osfrog, keepo, kappapride, elegiggle |
| Angka, rank, format tim | 4k, 3.8k, 1v1, 4v5, t2, 2k |
| Nama orang, tim, julukan | sumail, dendi, mineski, gaben, injoker, magina |
| Bahasa non-Inggris | cyka, blyat, bobo, kontol, puta, mierda, porfavor |
| Kata kasar baku | noob, retard, cunt, faggot, asshole |
| Pengulangan gabungan | wpwp, tyty, gogo, byebye |
| URL, emoticon, simbol | :d, t_t, www.leagueoflegends.com/register |

## Batas Leet

- Angka diubah jika hasilnya kata jelas: `h0w` -> `how`. `4k`, `t2` tetap.
- Leet fonetik ikut MultiLexNorm: `w8` -> `wait`, `gr8` -> `great`, `2day` -> `today`.

## Larangan

- Kata toksik tidak diperhalus atau dihapus: `cum` tetap `cum`.
- Bahasa lain tidak diterjemahkan.
- Akronim toksik tidak diekspansi: `wtf` tetap `wtf`.

## Riwayat

- v1 (2026-11-03): versi awal, batch TYPE-001.
- v2 (2026-11-04): kategori penyamaran.
