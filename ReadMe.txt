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



****Ta namestitev omogoča, da Flask aplikacija teče kot trajna storitev in ostane dostopna, tudi ko se odjavite iz SSH. *****

Zahteve
Strežnik z Ubuntu (npr. VM na Oracle Cloud).
Flask aplikacija v mapi (npr. /home/ubuntu/my-flask-app).
Gunicorn nameščen v virtualnem okolju.
Nginx nameščen na strežniku.


Korak 1: Konfiguracija Gunicorn kot systemd storitve
Ustvarite systemd storitveno datoteko za Gunicorn:

bash
sudo nano /etc/systemd/system/gunicorn.service

Dodajte naslednjo konfiguracijo:
Po potrebi zamenjajte poti in uporabniška imena:

ini
*code start*
[Unit]
Description=Gunicorn instance to serve Flask app
After=network.target

[Service]
User=ubuntu  # Zamenjajte z vašim uporabniškim imenom
Group=www-data
WorkingDirectory=/home/ubuntu/my-flask-app  # Pot do vaše aplikacije
Environment="PATH=/home/ubuntu/my-flask-app/venv/bin"  # Pot do vašega virtualnega okolja
ExecStart=/home/ubuntu/my-flask-app/venv/bin/gunicorn --workers 4 --timeout 120 -b 127.0.0.1:8080 wsgi:app

[Install]
WantedBy=multi-user.target
User: Uporabniško ime, pod katerim bo tekel Gunicorn.
WorkingDirectory: Pot do vaše Flask aplikacije.
Environment: Pot do bin mape v vašem virtualnem okolju.
ExecStart: Ukaz za zagon Gunicorn, ki posluša na 127.0.0.1:8080.
Zaženite in omogočite Gunicorn storitev:

bash
sudo systemctl daemon-reload
sudo systemctl start gunicorn
sudo systemctl enable gunicorn
Preverite stanje storitve:

bash
sudo systemctl status gunicorn
Ta ukaz bi moral pokazati, da je Gunicorn active (running).

Korak 2: Konfiguracija Nginx kot povratni posrednik (reverse proxy)
Ustvarite strežniški blok v Nginx za Flask aplikacijo:

bash
sudo nano /etc/nginx/sites-available/my-flask-app

Dodajte naslednjo konfiguracijo:

Zamenjajte 129.152.26.137 z vašim javnim IP naslovom ali domeno, če je na voljo.

nginx

server {
    listen 80;
    server_name 129.152.26.137;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}


Omogočite konfiguracijo:
Ustvarite simbolno povezavo do mape sites-enabled, da aktivirate konfiguracijo:

bash
sudo ln -s /etc/nginx/sites-available/my-flask-app /etc/nginx/sites-enabled/
Preizkusite in ponovno naložite Nginx:

Preizkusite konfiguracijo Nginx, da preverite, ali so prisotne kakšne napake v sintaksi:

bash
sudo nginx -t
Če ni napak, ponovno naložite Nginx, da uporabite spremembe:

bash
sudo systemctl reload nginx
Končno testiranje in odpravljanje napak
Preverite dostop: Obiščite http://129.152.26.137, da preverite, ali je vaša Flask aplikacija dostopna prek HTTP.

Preverite Gunicorn dnevnik za morebitne težave z aplikacijo:

bash
sudo journalctl -u gunicorn
Preverite Nginx dnevnik za težave s povezavo:

bash
sudo tail -f /var/log/nginx/error.log
S tem postopkom vaša Flask aplikacija teče kot storitev z Gunicorn in je dostopna prek Nginx-a na naslovu HTTP. Gunicorn bo še naprej deloval tudi po zaprtju SSH povezave, Nginx pa bo obravnaval dohodne HTTP zahteve in jih posredoval Gunicorn.

