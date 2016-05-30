import unittest
import DrakeBot as bot


class BotTests(unittest.TestCase):

    def test_no_quotes(self):
        str_1 = 'i hate drake'
        result = bot.remove_quoted_text(str_1)
        assert result == str_1

    def test_quoted_text(self):
        str_1 = 'i love drake "i hate drake"'
        str_2 = 'i love drake '
        result = bot.remove_quoted_text(str_1)
        assert result == str_2

    def text_mismatched_quotes(self):
        ''' If there are mismatched quotes in the input string, the
            remove_quoted_text() function shouldn't fuck with it.
            In this case we expect it to return the input string.
        '''
        str_1 = '""i hate drake"'
        result = bot.remove_quoted_text(str_1)
        assert result == str_1

    def test_load_oauth_keys(self):
        '''Tests that oauth keys are loaded correctly'''
        bot.load_oauth_keys()
        assert bot.CONSUMER_KEY

    def test_load_savepoint(self):
        '''Asserts last savepoint is not none'''
        savepoint = bot.load_savepoint()
        assert savepoint

    def test_twitter_search(self):
        '''Test that search functions as expected'''
        api = bot.twitter_oauth()
        tweets = bot.twitter_search(api, None)
        assert len(tweets) != 0

    def test_parse_savepoint(self):
        '''Test that a savepoint can be parsed from tweets'''
        api = bot.twitter_oauth()
        tweets = bot.twitter_search(api, None)
        savepoint = bot.parse_savepoint_from_tweets(tweets)
        assert savepoint  # is not an empty string

    def test_clean_search_results(self):
        '''Tests that cleaned results actually cleans some results'''
        api = bot.twitter_oauth()
        tweets = bot.twitter_search(api, None)
        cleaned = bot.clean_search_results(tweets)
        # this isn't necessarily true but in practice it
        # almost certainly will be
        assert len(cleaned) < len(tweets) and len(cleaned) != 0
