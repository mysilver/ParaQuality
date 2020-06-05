import language_tool_python

tool = language_tool_python.LanguageTool('en-US')


class GrammarError:
    def __init__(self, message, short_message, replacements, offset, length, rule):
        self.replacements = replacements
        self.offset = offset
        self.length = length
        self.rule = rule
        self.short_message = short_message
        self.message = message


class Client(object):

    def check(self, sentence):
        matches = tool.check(sentence)
        return [GrammarError(message=i.context,
                             replacements=i.replacements,
                             offset=i.offset,
                             length=i.errorLength,
                             rule=i.category,
                             short_message=i.message) for i in matches]


class LanguageChecker:
    def __init__(self):
        self.client = Client()

    def check(self, sentence, categories=None, excludes_ids=None):
        if not categories and not excludes_ids:
            return self.client.check(sentence)

        ret = []
        for error in self.client.check(sentence):
            if error.rule in categories and (
                    excludes_ids is None or error.rule not in excludes_ids):
                ret.append(error)

        return ret

    def misspellings(self, sentence):
        """
        :return: list of spelling errors 
        """
        resplacments = self.check(sentence, ['TYPOS'])
        ret = set()
        # corrections = set()
        for r in resplacments:
            ret.add(sentence[r.offset: r.offset + r.length])
            # if len(r.replacements) > 0:
            #     corrections.add(r.replacements[0]['value'])

        return ret

    def spelling_corrector(self, sentence):
        resplacments = self.check(sentence, ['TYPOS'])
        for r in resplacments:
            ms = sentence[r.offset: r.offset + r.length]
            if len(r.replacements) > 0:
                sentence = sentence.replace(ms, r.replacements[0])

        return sentence

    def grammar_corrector(self, sentence, categories=['MISC', 'GRAMMAR', "TYPOS"]):

        repls = self.check(sentence, categories=categories)

        while len(repls) > 0:
            r = repls[0]
            if not r.replacements:
                continue
            sentence = sentence[:r.offset] + r.replacements[0] + sentence[r.offset + r.length:]
            repls = self.check(sentence, categories=['MISC', 'GRAMMAR', 'TYPOS'])

        return sentence

    def singleWordCorrection(self, word):

        for p in self.check(word.replace('_', ' ')):
            if p.rule == 'TYPOS':
                for r in p.replacements:
                    if word in r:
                        return r.replace(' ', '_')
                if p.replacements:
                    return p.replacements[0].replace(' ', '_')
        return word

