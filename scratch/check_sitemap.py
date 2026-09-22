import urllib.request
try:
    print("ROBOTS.TXT")
    print(urllib.request.urlopen("https://www.fortunecloudindia.com/robots.txt").read().decode('utf-8'))
except Exception as e:
    print(e)

print("="*20)
try:
    print("SITEMAP.XML")
    print(urllib.request.urlopen("https://www.fortunecloudindia.com/sitemap.xml").read().decode('utf-8')[:500])
except Exception as e:
    print(e)
