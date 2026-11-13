# Domowe Archiwum Foto

Lokalna, szybka galeria i kataloger zdjęć oraz wideo dla systemu Windows, zaprojektowany do zarządzania dziesiątkami tysięcy plików rozproszonych na wielu zewnętrznych dyskach USB i pendrive'ach — **bez konieczności ich kopiowania**.

## Główne możliwości
- **Śledzenie dysków USB po numerze sprzętowym (Windows VSN)**: rozpoznaje pendrive'y i dyski zewnętrzne nawet po zmianie litery dysku w systemie Windows; wyświetla wolne miejsce (GB) i pozwala nadać dyskom własne krótkie nazwy.
- **Praca w trybie Offline**: generuje lekkie miniatury `WebP (300px)` na dysku lokalnym, dzięki czemu całą kolekcję można przeglądać, przeszukiwać i grupować nawet wtedy, gdy dyski USB są odłączone.
- **Wykrywanie duplikatów i anomalii dat**:
  - pełne hashowanie `SHA-256` dla zdjęć oraz szybkie hashowanie częściowe `3x64 KB` dla dużych plików wideo,
  - porównywanie klastrów duplikatów obok siebie ze wskazaniem oryginału (najstarszego pliku) i kopii,
  - wybór najstarszej wiarygodnej daty (`EXIF`, nazwa pliku, `ctime`/`mtime`) z flagowaniem rozbieżności $>24\text{h}$.
- **Lokalne AI i Computer Vision (On-Demand, bez obciążania RAM przy starcie)**:
  - tagowanie scen, twarzy i obiektów w języku polskim i angielskim (`OpenCV Haar Cascade` + `HSV Scene Analyzer` + sieć `ONNX MobileNetV2`),
  - detekcja zdjęć nieostrych/poruszonych na siatce `8x8` (odporna na szum ISO oraz gładkie niebo/bokeh),
  - **6 trybów wyszukiwania podobnych zdjęć** (Hybrydowe, Duplikaty wizualne `8x8 CIE LAB`, Te same obiekty AI, Paleta barw HSV, Ta sama sesja czasowa, Kompozycja i Twarze).
- **Operacje masowe i drzewiaste grupy**:
  - tworzenie hierarchicznego drzewa grup/podkatalogów (ręcznie lub 1-klikiem wg daty `Rok -> Miesiąc` oraz struktury folderów z dysku),
  - fizyczne przenoszenie i konsolidacja plików między dyskami USB,
  - odwracalny lokalny **Kosz (`data/kosz`)** z przywracaniem plików i całych grup na ich pierwotne miejsce.
- **Archiwum Google Drive API v3**: wysyłka zdjęć i kopii bazy SQLite do wybranej kartoteki na Google Drive z kryptograficzną weryfikacją sumy kontrolnej `MD5` oraz podglądem chmury na żywo.
- **Pełnoekranowy odtwarzacz i Tryb TV**: obsługa zdjęć (w tym `.HEIC`) i wideo (w tym `.MOV` z transkodowaniem w locie do `H.264 MP4`), nawigacja strzałkami z klawiatury/pilota oraz automatyczny pokaz slajdów na Google TV.

---

## Struktura katalogu `data/` (Start od zera)

Wszystkie dane robocze, baza SQLite, miniatury, modele AI oraz klucze chmurowe znajdują się w jednym katalogu `data/`:

```text
data/
├── db/
│   └── media_vault.db          # Tworzy się AUTOMATYCZNIE przy pierwszym starcie (tryb WAL)
├── backups/                    # Automatyczne kopie zapasowe bazy SQLite (przed operacjami plikowymi)
├── thumbnails/                 # Lokalne miniatury WebP (300px) oraz bufor strumieni MP4
├── exports/                    # Zapisane pliki eksportu metadanych (.json)
├── kosz/                       # Lokalny odwracalny kosz dla usuniętych zdjęć i grup
├── models/                     # Wagi modeli AI (pobierane skryptem poniżej)
│   ├── mobilenetv2.onnx        # Model klasyfikacji obiektów ONNX (~14 MB)
│   └── imagenet_classes.json   # Słownik 1000 klas ImageNet
├── credentials.json            # (Opcjonalnie) Klucz OAuth 2.0 Desktop z Google Cloud Console
└── token.json                  # Tworzy się AUTOMATYCZNIE po pierwszym zalogowaniu do Google Drive
```

## Jak przygotować folder data/ od zera (PowerShell)
Jeśli zaczynasz z czystym repozytorium (bez folderu data/), uruchom w głównym katalogu projektu poniższe polecenia w PowerShellu. 
Utworzą one wszystkie podkatalogi i pobiorą wymagane pliki modelu AI:

### 1. Utworzenie wszystkich wymaganych podkatalogów w `data/`

```code
New-Item -ItemType Directory -Force -Path `
    "data\db", `
    "data\backups", `
    "data\thumbnails", `
    "data\exports", `
    "data\kosz", `
    "data\models" | Out-Null
```
### 2. Pobranie słownika klas ImageNet (do tagowania AI)

```code
Invoke-WebRequest `
    -Uri "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json" `
    -OutFile "data\models\imagenet_classes.json"
```
### 3. Pobranie modelu sieci neuronowej ONNX MobileNetV2 (~14 MB z HuggingFace CDN)
```code
Invoke-WebRequest `
    -Uri "https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx" `
    -OutFile "data\models\mobilenetv2.onnx"

Write-Host "Katalog data/ oraz modele AI zostały pomyślnie przygotowane!" -ForegroundColor Green
```

## Instalacja środowiska i uruchomienie

### 1. Utworzenie i aktywacja środowiska wirtualnego Python 3.11+

```code
python -m venv venv
.\venv\Scripts\Activate.ps1```
```

### 2. Instalacja wymaganych bibliotek (wraz z obsługą HEIC i wideo MOV)

```code
pip install fastapi uvicorn[standard] sqlalchemy jinja2 python-multipart pillow pillow-heif opencv-python-headless imageio-ffmpeg numpy onnxruntime google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### 3. Uruchomienie serwera (lokalnie + dostęp w domowej sieci Wi-Fi dla TV)

```code
uvicorn main:app --host 0.0.0.0 --port 8000
```