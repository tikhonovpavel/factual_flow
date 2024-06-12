
class TokensToCheck:
    original_word: str
    target_word: str
    control_word: str

    original: str
    target: str
    control: str

    def __init__(self, original, target, control, add_space, tokenizer):
        self.original_word = original
        self.target_word = target
        self.control_word = control

        self.original = [TokensToCheck._get_token(x, add_space, tokenizer) for x in original] if isinstance(original, list) else TokensToCheck._get_token(original, add_space, tokenizer)
        self.target = [TokensToCheck._get_token(x, add_space, tokenizer) for x in target] if isinstance(target, list) else TokensToCheck._get_token(target, add_space, tokenizer)
        self.control = [TokensToCheck._get_token(x, add_space, tokenizer) for x in control] if isinstance(control, list) else TokensToCheck._get_token(control, add_space, tokenizer)

        self.add_space = add_space

    @staticmethod
    def _get_token(word, add_space, tokenizer):
        token = tokenizer.tokenize(' ' + word if add_space else word)[0]

        if tokenizer.tokenize(' ')[0] == token:
            token = tokenizer.tokenize(word)[0]

        return token

    def __repr__(self) -> str:
        return (f"TokensToCheck(original_word='{self.original_word}', target_word='{self.target_word}', control_word='{self.control_word}',\n"
                f"original='{self.original}', target='{self.target}', control='{self.control}')")

    def find_field_by_value(self, value) -> str:
        return next(key for key, val in self.__dict__.items() if val == value)

    def to_json(self):
        return {
            'original_word': self.original_word,
            'target_word': self.target_word,
            'control_word': self.control_word,
            'original': self.original,
            'target': self.target,
            'control': self.control
        }