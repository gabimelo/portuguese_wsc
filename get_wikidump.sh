if [ ! -f data/external/ptwiki-latest-pages-articles.xml.bz2 ]; then
    wget https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2  -P data/external
else
    echo "Wiki dump seems to already have been downloaded; will not download again"
fi
