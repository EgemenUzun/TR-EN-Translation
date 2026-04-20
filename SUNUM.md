# P14 — Sunum paketi (tek dosya)

Bu belge, **7 slaytlık sunumu** başka dosyaya ihtiyaç duymadan hazırlamanız için yazıldı. Aşağıdaki **tüm sayılar ve örnek metinler** proje test çıktılarından buraya taşınmıştır.

---

## Hızlı veri özeti (slayt 5’e yapıştırılabilir)

### Kaskad — MT karşılaştırması (100 örnek, ASR WER sabit)

Aynı koşuda **ASR WER = 21,37%** iken yalnızca MT değişmiştir:

| MT / koşu | ASR WER | BLEU | chrF |
|-----------|---------|------|------|
| Helsinki Opus-MT (taban) | 21,37% | 26,74 | 52,42 |
| Helsinki Opus-MT big | 21,37% | **30,51** | **55,63** |
| NLLB 1.3B | 21,37% | 16,06 | 36,96 |
| NLLB 600M (distilled) | 21,37% | 0,41 | 18,34 |

### Kaskad — diğer ölçekler

| Açıklama | N | ASR WER | BLEU | chrF |
|----------|---|---------|------|------|
| Alternatif 100 örnek koşusu | 100 | 33,98% | 21,66 | 47,58 |
| Geniş ölçek | 10 000 | 27,26% | 23,50 | 48,12 |

### Uçtan uca (E2E) — 100 örnek

| Metrik | Değer |
|--------|--------|
| BLEU | 0,00 |
| chrF | 0,00 |
| ASR WER | ölçülmedi (E2E’de Türkçe ara metin yok) |

### Ham log alıntıları (kanıt metni)

**Kaskad — 100 örnek (alternatif koşu):**
```
ASR WER (Word Error Rate) : 33.98% (Lower is better)
MT  BLEU Score            : 21.66 (Higher is better)
MT  chrF Score            : 47.58 (Higher is better)
```

**Kaskad — 10 000 örnek:**
```
ASR WER (Word Error Rate) : 27.26% (Lower is better)
MT  BLEU Score            : 23.50 (Higher is better)
MT  chrF Score            : 48.12 (Higher is better)
```

**Kaskad — Whisper large + Opus-MT taban (100 örnek):**
```
ASR WER (Word Error Rate) : 21.37% (Lower is better)
MT  BLEU Score            : 26.74 (Higher is better)
MT  chrF Score            : 52.42 (Higher is better)
```

**Kaskad — Whisper large + Opus-MT big (100 örnek):**
```
ASR WER (Word Error Rate) : 21.37% (Lower is better)
MT  BLEU Score            : 30.51 (Higher is better)
MT  chrF Score            : 55.63 (Higher is better)
```

**Kaskad — Whisper large + NLLB 1.3B (100 örnek):**
```
ASR WER (Word Error Rate) : 21.37% (Lower is better)
MT  BLEU Score            : 16.06 (Higher is better)
MT  chrF Score            : 36.96 (Higher is better)
```

**Kaskad — Whisper large + NLLB 600M (100 örnek):**
```
ASR WER (Word Error Rate) : 21.37% (Lower is better)
MT  BLEU Score            : 0.41 (Higher is better)
MT  chrF Score            : 18.34 (Higher is better)
```

**E2E değerlendirme (100 örnek):**
```
FULL END-TO-END SYSTEM EVALUATION RESULTS (100 Samples)
ASR WER (Word Error Rate) : N/A (E2E skips Turkish text!)
MT  BLEU Score            : 0.00 (Higher is better)
MT  chrF Score            : 0.00 (Higher is better)
```

---

## Slayt 1 — Başlık ve problem

**Başlık:** Türkçe→İngilizce konuşma çevirisi: kaskad ve uçtan uca  

**Alt başlık:** CoVoST2 + Common Voice • BLEU / chrF • ASR WER  

**Slayt metni:**
- **Hedef:** Türkçe sesi İngilizce metne çevirmek (speech translation).
- **Kaskad:** Ses → **ASR** (Türkçe) → **MT** (İngilizce).
- **Uçtan uca (E2E):** Ses → tek model → İngilizce (Wav2Vec2 + çeviri decoder).
- **Deney:** İki mimari; metrikler: **SacreBLEU**, **chrF**; kaskadda ek **WER**.

**Konuşmacı (15 sn):** Aynı veri hattında iki yaklaşımı ölçtük; kaskad için hem küçük hem büyük örneklem sonuçları var (100 ve 10 000 örnek).

---

## Slayt 2 — Veri ve protokol

**Başlık:** Veri seti ve kurulum  

**Slayt metni:**
- **Kaynak:** CoVoST2 (TR→EN); ses: Mozilla Common Voice (Türkçe).
- **Birleştirme:** `path` ile `validated.tsv` + CoVoST2 çeviri tablosu; diskte var olan `.mp3`/klipler tutuldu.
- **Çıktı:** Hugging Face `Dataset` — alanlar: `audio_path`, `sentence` (TR referans), `translation` (EN referans).
- **Metrikler:** İngilizce çıktı vs `translation` → BLEU, chrF; Türkçe ASR vs `sentence` → WER.
- **Ölçek:** Raporlanan koşular **N = 100** ve **N = 10 000**; ayrıca aynı WER altında **4 MT** karşılaştırması (hepsi N = 100).

**Tablo (protokol):**

| Öğe | Değer |
|-----|--------|
| Görev | Konuşma → İngilizce metin |
| ASR (kaskad) | Whisper (large-v3; bazı hızlı koşularda farklı varyant) |
| MT | Opus-MT tr-en (taban / big), NLLB 1.3B, NLLB 600M |
| E2E | wav2vec2-large-xlsr-53 + opus-mt-tr-en decoder, eğitim script’i ile |

---

## Slayt 3 — Kaskad (ASR → MT)

**Başlık:** Kaskad mimari  

**Diyagram (metin):** `Ses 16 kHz` → **Whisper** → Türkçe hipotez → **MT** → İngilizce  

**Slayt metni:**
- **Aşama 1 — ASR:** Türkçe transkripsiyon; dil `turkish`.
- **Aşama 2 — MT:** Seq2seq çeviri; model ailesine göre skorlar değişiyor.
- **Artı:** Modüler; WER ve çeviri ayrı raporlanır.
- **Eksi:** ASR hatası MT’ye aktarılır (hata zinciri).

**Sayısal vurgu (bu projeden):** WER **21,37%** sabitken BLEU **0,41** ile **30,51** arası — **MT ve ayarlar**, sabit ASR ile bile çeviriyi uçuruyor.

---

## Slayt 4 — Uçtan uca ST (E2E)

**Başlık:** Uçtan uca konuşma çevirisi  

**Slayt metni:**
- **Encoder:** `facebook/wav2vec2-large-xlsr-53`
- **Decoder:** `Helsinki-NLP/opus-mt-tr-en` (`SpeechEncoderDecoderModel`)
- **Eğitim:** Encoder üstü + decoder birlikte; feature encoder dondurulabilir; çıktı: `e2e_full_results/best_model`
- **Çıkarım:** `generate` (ör. beam); değerlendirme: 100 örnek, BLEU/chrF

**Kıyas tablosu:**

| | Kaskad | E2E |
|--|--------|-----|
| Ara metin | Var (Türkçe) | Yok |
| WER | Ölçülür | Tipik olarak raporlanmaz |
| Bu çalışmada EN kalitesi (100 örnek) | En iyi: BLEU **30,51**, chrF **55,63** | BLEU **0,00**, chrF **0,00** |

**Konuşmacı:** E2E metrikleri şu an sıfır; model yükleme / `lm_head` veya checkpoint sorunu olası — sunumda “ilerleme / düzeltme” olarak anlatılabilir.

---

## Slayt 5 — Sonuçlar (tüm tablolar)

**Başlık:** Deneysel sonuçlar  

**Tablo 1 — MT ablation (N=100, WER=21,37% sabit)**

| MT | BLEU | chrF |
|----|------|------|
| Opus-MT taban | 26,74 | 52,42 |
| Opus-MT big | **30,51** | **55,63** |
| NLLB 1.3B | 16,06 | 36,96 |
| NLLB 600M | 0,41 | 18,34 |

**Tablo 2 — Ölçek ve alternatif kaskad**

| Koşu | N | WER | BLEU | chrF |
|------|---|-----|------|------|
| Alternatif | 100 | 33,98% | 21,66 | 47,58 |
| Geniş | 10 000 | 27,26% | 23,50 | 48,12 |

**Tablo 3 — E2E (N=100)**

| BLEU | chrF |
|------|------|
| 0,00 | 0,00 |

**Ana mesaj:** (1) Kaskadda **en iyi** ölçülen çeviri: **BLEU 30,51**, **chrF 55,63** (Whisper large + Opus-MT big, 100 örnek). (2) **10 000** örnekte **WER 27,26%**, **BLEU 23,50**, **chrF 48,12**. (3) **WER–çeviri:** Aynı WER’de MT değişimi BLEU’yu **0,41**’den **30,51**’e çıkarıyor — çeviri kalitesi yalnızca WER’e değil, **MT kalitesine ve protokole** bağlı.

---

## Slayt 6 — Hata analizi (tam örnek metinleri)

**Başlık:** ASR hataları: özel isim, anlam kayması, model boyutu  

### A) Whisper large-v3 — örnekler (ilk 100 test döngüsünden)

| # | Referans (TR) | Whisper çıktısı (TR) |
|---|----------------|----------------------|
| 1 | Seydiu şimdi iki mevkiyi de kaybetti. | Seydiu şimdiki mevkii de kaybetti. |
| 2 | Siyasette temiz insanlara ihtiyacımız var. | Siyasette temiz insanlara ihtiyacımız var. |
| 3 | Broz, büyükbabasının başarılarını gururla anıyor. | Broz, büyük babasının başarılarını gururla anıyor. |
| 4 | Calasan yetenekli çocuk bursu da aldı. | Calasım yetenekli çocuk bursu da aldı. |
| 7 | Atık yönetimi bir ülkenin yaşam tarzını yansıtır. | Atatürk öğretmeni ve ülkenin yaşam tarzında yansıttı. |
| 8 | Hükümetin değişmesi halinde müzakereler ve katılım süreci ne yönde etkilenir? | (aynı) |
| 14 | Biz realiteyi dikkate alıyoruz. | icraat şeyi dikkate alıyoruz. |
| 20 | Erdoğan eleştirileri reddetti. | Erdoğan eleştirileri reddetti. |
| 25 | Ve bir yıl sonra, bu ay, bu iki kitap basıldı. | Ve dediğim sonradan bu ay bu iki kitap basıldı. |

**Yorum (slayta 2 madde):**
- **Özel isim / varlık:** «Seydiu», «Broz», «Erdoğan» — bazen korunuyor, bazen bozuluyor.
- **Ciddi anlam hatası:** Örnek 7 — çöp yönetimi cümlesi **tamamen farklı** bir içeriğe dönüşmüş; kaskadda MT bu hatayı düzeltemez (yanlış kaynak metin).
- **Uzun soru (ör. #8):** Bu örnekte Whisper **satır satır doğru**; uzun cümle her zaman zor değil, ancak #7 ve #14 kısa cümlelerde bile ciddi hata var.

### B) Daha küçük Whisper — `small_out` (3 örnek, ham metin)

```
Örnek 1:
Gerçekte Söylenen : Seydiu şimdi iki mevkiyi de kaybetti.
Whisper'ın Duyduğu: Seydiği, şimdi iki mevkide kaybetti.
----------------------------------------
Örnek 2:
Gerçekte Söylenen : Siyasette temiz insanlara ihtiyacımız var.
Whisper'ın Duyduğu: Siyasette temiz insanlara ihtiyacımız var.
----------------------------------------
Örnek 3:
Gerçekte Söylenen : Broz, büyükbabasının başarılarını gururla anıyor.
Whisper'ın Duyduğu: Broz büyük babasının başarılarını gururla anıyor.
```

### C) Daha küçük Whisper — `tiny_out` (3 örnek, ham metin)

```
Örnek 1:
Gerçekte Söylenen : Seydiu şimdi iki mevkiyi de kaybetti.
Whisper'ın Duyduğu: Sevdiyor, şimdi iki mevkide kaybettiği.
----------------------------------------
Örnek 2:
Gerçekte Söylenen : Siyasette temiz insanlara ihtiyacımız var.
Whisper'ın Duyduğu: Siyası tetimizin sonu bekliyordun süreceğim.
----------------------------------------
Örnek 3:
Gerçekte Söylenen : Broz, büyükbabasının başarılarını gururla anıyor.
Whisper'ın Duyduğu: Burası, büyük babasının başlarlarını grurla anıyor.
```

**Slayt kapanış cümlesi:** Küçük modelde **#2 ve #3** çöküyor; large-v3’te birçok örnek düzeliyor ama **#7** gibi nadir ama şiddetli hatalar kalıyor — **halüsinasyon riski** için etik slaytına köprü.

---

## Slayt 7 — Sonuç, risk, etik, okuma

**Başlık:** Özet ve sorumlu kullanım  

**Sonuç maddeleri (rakamlarla):**
- Kaskad **en iyi:** BLEU **30,51**, chrF **55,63**, WER **21,37%** (100 örnek; Opus-MT big).
- Kaskad **büyük ölçek:** WER **27,26%**, BLEU **23,50**, chrF **48,12** (10 000 örnek).
- **MT etkisi:** Sabit WER’de BLEU **0,41 ↔ 30,51**.
- E2E: **BLEU/chrF 0,00** — değerlendirme veya model yükleme iyileştirmesi gerekli.

**Risk ve etik:**
- Otomatik çeviri **yanlış veya uydurma** içerik üretebilir (örnek: «Atık yönetimi» → «Atatürk öğretmeni»).
- Kritik alanlarda insan denetimi; sistem tek başına karar verdirilmemeli.
- Veri ve modeller: Common Voice, CoVoST2, Whisper, Wav2Vec2, Opus-MT, NLLB lisansları makalede atılmalı.

**Okuma listesi (8–12):**
1. CoVoST2 (Wang et al.)  
2. Whisper raporu  
3. Konuşma çevirisi / ST mimarileri (survey veya klasik ST makalesi)  
4. SacreBLEU  
5. chrF (Popović)  
6. WER  
7. Common Voice / FLEURS dokümantasyonu  
8. Wav2Vec2-XLSR  
9. Marian / Opus-MT  
10. NLLB  
11. NLP güvenilirliği ve halüsinasyon  
12. Topluluk ses verisi etiği  

**Kapanış:** Sorular.

---

## Sunumu üretirken

- Slayt 5 için yukarıdaki **“Hızlı veri özeti”** veya **Slayt 5 tabloları** doğrudan kopyalanabilir.
- Slayt 6 için tabloları veya ham kutuları seçerek yapıştırın; çok kalabalıksa yalnızca **#7, #14, tiny #2** ile 3 örnek yeter.
- Tüm ölçümler yerel GPU koşularından alınmıştır; raporda “hardware / sürüm” tek cümle eklenebilir.
