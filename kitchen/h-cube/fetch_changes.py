#!/usr/bin/env python3
import os
import argparse
import sys
import json
import requests
import json

from lxml import html


def fetch_news_pages(pkg):
    r = requests.get(f"https://tracker.debian.org/pkg/{pkg}/news")
    r.raise_for_status();
    pages = [html.fromstring(r.content)]

    page_count = int(pages[0].xpath('count(//ul[@class="pagination"]/li)'))
    while len(pages) < page_count:
        r = requests.get(f"https://tracker.debian.org/pkg/{pkg}/news", params=[
            ("page", len(pages) + 1)
        ])
        r.raise_for_status();
        pages.append(html.fromstring(r.content))

    return pages


def parse_news_page(page):
    for entry in page.xpath('//li[@class="list-group-item"]'):
        date = entry.xpath('.//span[@class="news-date"]/text()')[0]
        title = entry.xpath('.//span[@class="news-title"]/text()')[0]
        link = entry.xpath('.//a[starts-with(@href, "/news/")]/@href')[0]
        yield { 'date': date, 'title': title, 'link': link }


def main(args):
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, args.package)
    if os.path.exists(out_path):
        return

    news = [
        item
        for page in fetch_news_pages(args.package)
        for item in parse_news_page(page)
    ]

    with open(out_path, "w") as f:
        json.dump(news, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch tracker.debian.org news entries for the given package")
    parser.add_argument("out", metavar='DIR', help="Output directory")
    parser.add_argument("package", metavar='PKG', help="Package name")
    main(parser.parse_args())