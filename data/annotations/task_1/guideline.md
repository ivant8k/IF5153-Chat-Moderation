# Guideline Normalisasi Task 1

Versi: v2 (2026-11-04)
Dipakai oleh: pelabelan LLM (lapis 6, 7) dan QC manusia (review_mln, llm_types, gold).

## Esensi

Task 1 harus mengembalikan kata yang sengaja disamarkan pemain ke bentuk aslinya, untuk semua kata, bukan hanya kata kasar.

## Unit

- Satu slot = satu token raw. Pengecualian: 3 huruf tunggal atau lebih berurutan (`f u c k`) digabung jadi satu slot oleh tokenizer.
- Output per slot: lowercase, boleh lebih dari satu kata.
- Slot tidak boleh digabung, dipecah, atau dikosongkan.
- Target: bahasa Inggris baku, mengikuti gaya MultiLexNorm.
- Ragu -> biarkan. Over-normalization (R-01) lebih berbahaya daripada under-normalization.

## Diubah

| Jenis | Contoh |
| --- | --- |
| Leet (angka sebagai huruf) | b1tch -> bitch, h0w -> how, n00b -> noob, 5h1t -> shit, a55 -> ass |
| Simbol sebagai huruf | $tupid -> stupid, !diot -> idiot, d@mn -> damn, sh!t -> shit, jacka$$ -> jackass |
| Simbol penyisip | f*ck -> fuck, f.u.c.k -> fuck, b-i-t-c-h -> bitch |
| Campuran leet + huruf berulang | y0o0o -> yo, n0000b -> noob, fuuuck -> fuck |
| Huruf dipisah spasi (satu slot) | f u c k -> fuck, n i k i t a -> nikita, u r a -> you are a |
| Aksen, homoglyph | köll -> kill, b¡tch -> bitch, ßig -> big |
| Huruf hilang yang disengaja | idot -> idiot, stupd -> stupid, fck -> fuck |
| Typo, huruf tertukar | jsut -> just, reprot -> report, gmae -> game, teh -> the |
| Singkatan chat yang punya padanan baku | sry -> sorry, rly -> really, tnx -> thanks, dunno -> don't know, w/e -> whatever |
| Apostrof hilang | shouldnt -> shouldn't, ure -> you're, i`m -> i'm |
| Huruf berulang ekspresif | nooob -> noob, noooo -> no, plss -> please |
| Leet fonetik | w0w -> wow, gr8 -> great, m8 -> mate, n1 -> nice one |
| Kata kasar salah eja | fking -> fucking, fuckign -> fucking, sht -> shit, tard -> retard |
| Varian istilah game -> bentuk di lexicon | izi -> ez, gege -> gg, wpo -> wp, supp -> support |
| Nama hero/item salah eja | zues -> zeus, pheonix -> phoenix, meppo -> meepo |
| Singkatan satuan | lvl -> level, dmg -> damage, secs -> seconds, 2mins -> 2 minutes |

## Dibiarkan

| Jenis | Contoh |
| --- | --- |
| Istilah game di lexicon CONDA (lapis 1) | wk, sf, bkb, mid, gg, ez |
| Singkatan hero, role, mekanik tanpa padanan umum | ns, ls, ud, fv, gpm, int, bg, hf |
| Akronim yang dibiarkan MultiLexNorm | wtf, stfu, idk, omg, lol, ffs, tfw |
| Tawa, interjeksi | jaja, kek, lel, xaxa, wahaha, meh, ouch, shh, pff |
| Emote Twitch | 4head, osfrog, keepo, kappapride, elegiggle |
| Angka, rank, format tim (bukan leet) | 4k, 3.8k, 1v1, 4v5, t2, 2k |
| Nama orang, tim, julukan | sumail, dendi, mineski, gaben, injoker, magina |
| Bahasa non-Inggris | cyka, blyat, bobo, kontol, puta, mierda, porfavor |
| Kata kasar baku | noob, retard, cunt, faggot, asshole |
| Pengulangan gabungan | wpwp, tyty, gogo, byebye |
| URL, emoticon, simbol | :d, t_t, www.leagueoflegends.com/register |

## Batas Leet

- Angka diubah hanya jika hasilnya kata yang jelas: `h0w` -> `how`, tapi `4k` dan `t2` tetap.
- Leet fonetik ikut aturan MLN: `w8` -> `wait`, `gr8` -> `great`, `2day` -> `today`.

## Larangan

- Kata toksik tidak boleh diperhalus atau dihilangkan: `cum` tetap `cum`.
- Jangan menerjemahkan bahasa lain.
- Jangan mengekspansi akronim toksik (`wtf`). Ekspansi dilakukan kamus terpisah setelah T5 (D1).

## Riwayat

- v1 (2026-11-03): versi awal, lapis 6 batch L6-001.
- v2 (2026-11-04): tambah esensi dan kategori penyamaran disengaja (keputusan user D7).
