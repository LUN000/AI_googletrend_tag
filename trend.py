from pytrends.request import TrendReq
from datetime import date

class trend:
    def __init__(self, kw_list=["food", "art", "travel", "hotel"], timeframe='today 12-m', keywords_geo='US', daily_geo='united_states'):
        self.kw_list = kw_list
        self.timeframe = timeframe
        self.keywords_geo = keywords_geo
        self.daily_geo = daily_geo

    def get_trend(self):
        pytrend = TrendReq(hl='en-US', tz=360)

        key_trending = {}
        pytrend.build_payload(self.kw_list, timeframe=self.timeframe, geo=self.keywords_geo)  # timeframe 可選取時間區間, geo 可選取搜尋地點
        topics = pytrend.related_topics()
        queries = pytrend.related_queries()

        for kw in self.kw_list:
            try:
                key_trending.update({kw:{
                        "top_topics": topics[kw]["top"].to_dict()["topic_title"],
                        "rising_topics": topics[kw]["rising"].to_dict()["topic_title"],
                        "top_queries": queries[kw]["top"].to_dict()["query"],
                        "rising_queries": queries[kw]["rising"].to_dict()["query"]
                    }})
            except:
                key_trending.update({kw:{
                        "top_queries": queries[kw]["top"].to_dict()["query"],
                        "rising_queries": queries[kw]["rising"].to_dict()["query"]
                    }})

        daily_trending = pytrend.trending_searches(pn=self.daily_geo).to_dict()[0] # 24hr 搜尋數高字詞
        realtime_trending = pytrend.realtime_trending_searches(count=20).title.to_dict()  # 24hr 上升趨勢快的事件關鍵字

        result = {"key_trending": key_trending, "daily_trending":daily_trending, "realtime_trending":realtime_trending, }
        return result

    def get_tags(self):
        trend_dict = self.get_trend()
        tags = []
        for kw in self.kw_list:
            tags += trend_dict['key_trending'][kw]['top_queries'].values()
            tags += trend_dict['key_trending'][kw]['rising_queries'].values()
        tags += trend_dict["daily_trending"].values()
        tags += trend_dict["realtime_trending"].values()
        return tags
