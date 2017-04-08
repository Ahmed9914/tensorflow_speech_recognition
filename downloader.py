import urllib
import os
import sys

base = "http://www.everyayah.com/data"
reciters = [
    "AbdulSamad_64kbps_QuranExplorer.Com",
    "Abdul_Basit_Murattal_64kbps",
    "Abdullaah_3awwaad_Al-Juhaynee_128kbps",
    "Abdullah_Basfar_64kbps",
    "Abdullah_Matroud_128kbps",
    "Abdurrahmaan_As-Sudais_64kbps",
    "Abu_Bakr_Ash-Shaatree_64kbps",
    "Ahmed_Neana_128kbps",
    "Ahmed_ibn_Ali_al-Ajamy_64kbps_QuranExplorer.Com",
    "Akram_AlAlaqimy_128kbps",
    "Alafasy_64kbps",
    "Ali_Hajjaj_AlSuesy_128kbps",
    "Ali_Jaber_64kbps",
    "Fares_Abbad_64kbps",
    "Ghamadi_40kbps",
    "Hani_Rifai_64kbps",
    "Hudhaify_64kbps",
    "Husary_64kbps",
    "Ibrahim_Akhdar_32kbps",
    "Karim_Mansoori_40kbps",
    "Khaalid_Abdullaah_al-Qahtaanee_192kbps",
    "Maher_AlMuaiqly_64kbps",
    "Menshawi_32kbps",
    "Minshawy_Murattal_128kbps",
    "Mohammad_al_Tablaway_64kbps",
    "Muhammad_AbdulKareem_128kbps",
    "Muhammad_Ayyoub_64kbps",
    "Muhammad_Jibreel_64kbps",
    "Muhsin_Al_Qasim_192kbps",
    "Mustafa_Ismail_48kbps",
    "Nasser_Alqatami_128kbps",
    "Parhizgar_48kbps",
    "Sahl_Yassin_128kbps",
    "Salaah_AbdulRahman_Bukhatir_128kbps",
    "Salah_Al_Budair_128kbps",
    "Saood_ash-Shuraym_64kbps",
    "Yaser_Salamah_128kbps",
    "Yasser_Ad-Dussary_128kbps",
]

if len(sys.argv) < 2:
    print "For example, downloading Al-Fatihah ayah 1:\npython %s 001 001" % sys.argv[0]
    sys.exit(1)

surah = sys.argv[1]
ayah  = sys.argv[2]

directory = "wav/train/" + surah
if not os.path.exists(directory):
    os.makedirs(directory)

directory += "/" + ayah
if not os.path.exists(directory):
    os.makedirs(directory)

for i, r in enumerate(reciters):
    dl = base + "/" + r + "/" + surah + ayah + ".mp3"
    loc = directory + "/" + surah + ayah + "_{:02d}".format(i) + ".mp3"
    print "[%2d/%2d] Downloading '%s' to '%s'" % (i+1, len(reciters), dl, loc)
    urllib.urlretrieve(dl, loc)


