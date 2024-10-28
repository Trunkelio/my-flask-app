Ta Flask API omogoča prepoznavanje vrst rastlin, analizo zdravstvenega stanja rastlin in diagnozo bolezni specifičnih za posamezne vrste. API uporablja predtrenirane modele strojnega učenja, ki so zasnovani v Orange Data Mining in naloženi v aplikacijo prek pickle datotek.

Arhitektura
API uporablja naslednje komponente:

Flask: Glavni spletni okvir za obravnavanje HTTP zahtevkov in upravljanje API končnih točk.
TorchVision: Za pridobivanje značilk iz slik (embeddingov) z uporabo predtrenirane mreže SqueezeNet.
Orange Data Mining: Uporabljeni modeli za prepoznavanje vrste rastline, analizo zdravja in diagnozo bolezni so bili trenirani v okolju Orange in nato shranjeni kot pickle datoteke.
Nginx: Povzema zahteve na port 80 in jih posreduje Gunicornu na 127.0.0.1:8080.
Gunicorn: WSGI strežnik, ki omogoča zanesljivo izvajanje Flask aplikacije.

Glavne funkcionalnosti API-ja

API zagotavlja tri glavne končne točke:
/health: Prepoznavanje zdravstvenega stanja rastline (zdrava ali bolna).
/species: Prepoznavanje vrste rastline.
/sickness: Prepoznavanje bolezni, specifične za prepoznano vrsto rastline.

Zahteve
API uporablja naslednje knjižnice, ki jih najdete v requirements.txt datoteki. Nekatere ključne odvisnosti vključujejo:
Flask: Mikro okvir za ustvarjanje spletnega API-ja.
Orange3: Platforma za podatkovno rudarjenje in strojno učenje.
TorchVision: Za delo z vnaprej usposobljenimi modeli za ekstrakcijo značilk iz slik.
Pillow: Za delo z obdelavo slik.
Nginx in Gunicorn: Za upravljanje strežnika in obdelavo zahtev.

Konfiguracija Nginx in Gunicorn
Nginx posluša na portu 80 in preusmerja vse zahteve na Gunicorn strežnik, ki izvaja Flask aplikacijo na 127.0.0.1:8080.
Nginx konfiguracija (v /etc/nginx/sites-available/default):

nginx
Copy code
server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
Zagon Gunicorn:

bash
Copy code
gunicorn -w 4 -b 127.0.0.1:8080 app:app
Parametri: -w 4 pomeni štiri delovne procese, kar omogoča boljšo porazdelitev prometa.

API Končne točke
1. /health - Klasifikacija zdravstvenega stanja rastline
Metoda: POST
Opis: Sprejme sliko rastline in analizira zdravstveno stanje (npr. zdrava ali bolna).

Primer zahtevka:
bash
curl -X POST -F file=@path/to/plant_image.jpg http://example.com/health

Odgovor:
json
{
  "health_status": "Zdrav"
}

2. /species - Klasifikacija vrste rastline
Metoda: POST
Opis: Sprejme sliko rastline, prepozna njeno vrsto in vrne ime vrste ter stopnjo zaupanja.

Primer zahtevka:
bash
curl -X POST -F file=@path/to/plant_image.jpg http://example.com/species

Odgovor:
json
Copy code
{
  "species_name": "Jabolko",
  "confidence": 92.5
}

3. /sickness - Diagnoza bolezni glede na vrsto rastline
Metoda: POST
Opis: Najprej prepozna vrsto rastline, nato uporabi ustrezen model za prepoznavanje bolezni, specifične za prepoznano vrsto.

Primer zahtevka:
bash
curl -X POST -F file=@path/to/plant_image.jpg http://example.com/sickness

Odgovor:
json
{
  "species_name": "Jabolko",
  "species_confidence": 92.5,
  "disease": "Siva pečka",
  "disease_confidence": 88.3
}

Potek Delovanja
Nalaganje Modelov: Ob zagonu aplikacije se naložijo vsi potrebni modeli (.pkcls datoteke) za klasifikacijo zdravja, prepoznavanje vrst in bolezni.
Pridobivanje značilk: Za vsako poslano sliko se pridobijo značilke (embeddingi) z uporabo predtrenirane SqueezeNet mreže.
Pretvorba v Orange format: Pridobljene značilke se pretvorijo v tabelo, ki je združljiva z Orange modeli.
Klasifikacija: Glede na končno točko (/health, /species, /sickness) API izvede ustrezno klasifikacijo in vrne rezultat.
Posredovanje Rezultatov: Rezultat se vrne kot JSON odgovor, ki vključuje zdravstveno stanje, ime vrste in (če je ustrezno) ime bolezni ter stopnjo zaupanja.


Namestitev in zagon
Klonirajte repozitorij in se premaknite v direktorij projekta.

bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo

Namestite odvisnosti:
bash
pip install -r requirements.txt
Zagon aplikacije lokalno:

bash
python app.py

Zagon s Gunicorn in Nginx:

Zaženite Gunicorn:
bash
gunicorn -w 4 -b 127.0.0.1:8080 app:app
Nastavite Nginx, kot je opisano zgoraj, da preusmerja zahteve na Gunicorn.

Opombe
Prepričajte se, da imate nameščene vse zahtevane Python pakete, kot je navedeno v requirements.txt.
Če strežnik gosti več aplikacij, preverite nastavitve Nginx in Gunicorn, da se izognete konfliktom.
app.py privzeto uporablja CPU za vse izračune. Če imate na voljo GPU in potrebujete optimizacijo, poskrbite za ustrezno prilagoditev v klicih torch.
