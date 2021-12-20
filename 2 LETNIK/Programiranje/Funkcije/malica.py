# =============================================================================
# Malica
#
# Srednje šole po Sloveniji si prizadevajo, da bi njihovi dijaki jedli
# čim bolj zdravo malico. Ravnatelj ene od srednjih šol je vaš dobri
# prijatelj. Želi, da mu napišete program, s katerim bo lahko analiziral
# jedilnike. Želi si, da bi bili njegovi dijaki zdravi in srečni.
# 
# Pri tej nalogi si lahko pomagate z operatorjema `in` in `not in`:
# 
#     >>> 2 in [4, 5, 2, 7]
#     True
#     >>> 3 in [4, 5, 2, 7]
#     False
#     >>> 3 not in [4, 5, 2, 7]
#     True
# =====================================================================@005676=
# 1. podnaloga
# Jedilnik opišemo s seznamom nizov, kot vidite na primeru v preambuli.
# Vsak zaporedni element seznama ustreza enemu od zaporednih dni.
# Jedilnik se periodično ponavlja. Če je dolžina jedilnika `n`, bodo
# dijaki $(n+1)$-ti dan spet jedli tisto, kar je na prvem mestu na
# jedilniku. Ravnatelj je opazil, da so nekatere stvari na jedilniku
# napisane po večkrat. Rad bi, da napišete funkcijo `brez_ponovitev(l)`,
# ki sestavi nov seznam, tako da se bo vsak element pojavil samo enkrat.
# 
#     >>> brez_ponovitev(primer_jedilnika)
#     ['svinjski zrezek v omaki', 'sirov kanelon', 'ocvrt oslič', 'sirov burek']
# 
# Elementi naj se v novem seznamu pojavijo v takem vrstnem redu, kot
# njihove prve ponovitve v originalnem seznamu `l`.
# =============================================================================
primer_jedilnika = [
    'svinjski zrezek v omaki', 'sirov kanelon', 'ocvrt oslič',
    'svinjski zrezek v omaki', 'ocvrt oslič', 'sirov burek',
    'sirov kanelon', 'ocvrt oslič', 'sirov kanelon', 'sirov kanelon'
]
def brez_ponovitev(primer_jedilnika):
    brez_ponav=[]
    for i in primer_jedilnika:
        if i not in brez_ponav:
            brez_ponav.append(i)
    return brez_ponav
# =====================================================================@005677=
# 2. podnaloga
# Ravnatelj je sicer ugotovil, da se jedi ponavljajo. Če pa je od
# dneva, ko je bila neka jed nazadnje na jedilniku, minilo dovolj časa,
# ni s tem nič narobe. Napišite funkcijo `kdaj_prej(l)`, ki dobi nek
# jedilnik in vrne enako dolg seznam, kjer je za vsak dan ena številka in
# sicer koliko dni je minilo od takrat, ko je bila jed nazadnje na
# jedilniku. Upoštevajte, da je jedilnik periodičen. Primer:
# 
#     >>> kdaj_prej(['burek', 'jogurt', 'burek'])
#     [1, 3, 2]
#     >>> kdaj_prej(primer_jedilnika)
#     [7, 2, 5, 3, 2, 10, 5, 3, 2, 1]
# =============================================================================
def kdaj_prej(l):
    ponavlja=[]
    n=len(l)
    for i in range(n):
        p=1
        j=(i-1)
        while l[j % n] != l[i]:
            p+=1
            j-=1
        ponavlja.append(p)
    return ponavlja

# =====================================================================@005678=
# 3. podnaloga
# Ravnatelj je definiral pojma _raznolikost_ in _kakovost_. (Opomba:
# Ravnatelj je študiral fiziko in jo svoje čase tudi poučeval.)
# _Raznolikost_ je število različnih jedi na jedilniku. _Kakovost_ pa
# je povprečje kvadratov vrednosti elementov seznama, ki ga vrne funkcija
# `kdaj_prej(l)`. Sestavite funkcijo `raznolikost_in_kakovost(l)`, ki
# vrne par števil in sicer raznolikost in kakovost jedilnika `l`. Primer:
# 
#     >>> raznolikost_in_kakovost(['burek', 'burek', 'jogurt'])
#     (2, 4.666666666666667)
#     >>> raznolikost_in_kakovost(primer_jedilnika)
#     (4, 23.0)
# =============================================================================
def raznolikost_in_kakovost(l):
    raznolikost=brez_ponovitev(l)
    kakovost= [ x**2 for x in kdaj_prej(l)]
    povprecje= sum(kakovost) / len(kakovost)
    return (len(raznolikost), povprecje)


        
# =====================================================================@005679=
# 4. podnaloga
# Ravnatelj je v zbornici zbral različne predloge jedilnikov in jih
# združil v en sam seznam. Sestavite funkcijo `naj_jedilnik(ll)`, ki
# izmed vseh teh jedilnikov v seznamu `ll` izbere in vrne najboljšega.
# Bolši je tisti jedilnik, ki je bolj raznolik. Med jedilniki z enako
# raznolikostjo je boljši tisti, ki je kakovostnejši. Predpostavite, da
# v podatkih ne bo dveh enako dobrih jedilnikov. Primer:
# 
#     >>> naj_jedilnik([['burek', 'jogurt'], ['pomaranča', 'pomaranča']])
#     ['burek', 'jogurt']
#     >>> naj_jedilnik([['burek', 'jogurt'], primer_jedilnika])
#     ['svinjski zrezek v omaki', 'sirov kanelon', 'ocvrt oslič',
#     'svinjski zrezek v omaki', 'ocvrt oslič', 'sirov burek',
#     'sirov kanelon', 'ocvrt oslič', 'sirov kanelon', 'sirov kanelon']
# =============================================================================

def naj_jedilnik(ll):
    naj_jedilnik= None
    for i in ll:
        if naj_jedilnik is None:
            naj_jedilnik=i 
        elif  raznolikost_in_kakovost(i) > raznolikost_in_kakovost(naj_jedilnik):
            naj_jedilnik=i
    return naj_jedilnik




































































































# ============================================================================@

'Če vam Python sporoča, da je v tej vrstici sintaktična napaka,'
'se napaka v resnici skriva v zadnjih vrsticah vaše kode.'

'Kode od tu naprej NE SPREMINJAJTE!'


















































import json, os, re, sys, shutil, traceback, urllib.error, urllib.request


import io, sys
from contextlib import contextmanager

class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end='')
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end='')
        return line

class Check:
    @staticmethod
    def has_solution(part):
        return part['solution'].strip() != ''

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part['valid'] = True
            part['feedback'] = []
            part['secret'] = []
        Check.current_part = None
        Check.part_counter = None

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part['feedback'].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part['valid'] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed))
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted([(Check.clean(k, digits, typed), Check.clean(v, digits, typed)) for (k, v) in x.items()])
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get('clean', clean)
        Check.current_part['secret'].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error('Izraz {0} vrne {1!r} namesto {2!r}.',
                        expression, actual_result, expected_result)
            return False
        else:
            return True

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        exec(code, global_env)
        errors = []
        for (x, v) in expected_state.items():
            if x not in global_env:
                errors.append('morajo nastaviti spremenljivko {0}, vendar je ne'.format(x))
            elif clean(global_env[x]) != clean(v):
                errors.append('nastavijo {0} na {1!r} namesto na {2!r}'.format(x, global_env[x], v))
        if errors:
            Check.error('Ukazi\n{0}\n{1}.', statements,  ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get('encoding', encoding)
        with open(filename, 'w', encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part['feedback'][:]
        yield
        new_feedback = Check.current_part['feedback'][len(old_feedback):]
        Check.current_part['feedback'] = old_feedback
        if new_feedback:
            new_feedback = ['\n    '.join(error.split('\n')) for error in new_feedback]
            Check.error('Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}', filename, '\n  '.join(content), '\n- '.join(new_feedback))

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part['feedback'][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get('stringio')('\n'.join(content) + '\n')
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part['feedback'][len(old_feedback):]
        Check.current_part['feedback'] = old_feedback
        if new_feedback:
            new_feedback = ['\n  '.join(error.split('\n')) for error in new_feedback]
            Check.error('Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}', '\n  '.join(content), '\n- '.join(new_feedback))

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get('encoding', encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error('Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}', filename, (line_width - 7) * ' ', '\n  '.join(diff))
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(expression, global_env)
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal:
            return True
        else:
            Check.error('Program izpiše{0}  namesto:\n  {1}', (line_width - 13) * ' ', '\n  '.join(diff))
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ['\n']
        else:
            expected_lines += (actual_len - expected_len) * ['\n']
        equal = True
        line_width = max(len(actual_line.rstrip()) for actual_line in actual_lines + ['Program izpiše'])
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append('{0} {1} {2}'.format(out.ljust(line_width), '|' if out == given else '*', given))
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get('update_env', update_env):
            global_env = dict(global_env)
        global_env.update(Check.get('env', env))
        return global_env

    @staticmethod
    def generator(expression, expected_values, should_stop=None, further_iter=None, clean=None, env=None, update_env=None):
        from types import GeneratorType
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error("Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
                                iteration, expression, actual_value, expected_value)
                    return False
            for _ in range(Check.get('further_iter', further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False
        
        if Check.get('should_stop', should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print('{0}. podnaloga je brez rešitve.'.format(i + 1))
            elif not part['valid']:
                print('{0}. podnaloga nima veljavne rešitve.'.format(i + 1))
            else:
                print('{0}. podnaloga ima veljavno rešitev.'.format(i + 1))
            for message in part['feedback']:
                print('  - {0}'.format('\n    '.join(message.splitlines())))

    settings_stack = [{
        'clean': clean.__func__,
        'encoding': None,
        'env': {},
        'further_iter': 0,
        'should_stop': False,
        'stringio': VisibleStringIO,
        'update_env': False,
    }]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs))
                             if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get('env'))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get('stringio'):
            yield
        else:
            with Check.set(stringio=stringio):
                yield


def _validate_current_file():
    def extract_parts(filename):
        with open(filename, encoding='utf-8') as f:
            source = f.read()
        part_regex = re.compile(
            r'# =+@(?P<part>\d+)=\s*\n' # beginning of header
            r'(\s*#( [^\n]*)?\n)+?'     # description
            r'\s*# =+\s*?\n'            # end of header
            r'(?P<solution>.*?)'        # solution
            r'(?=\n\s*# =+@)',          # beginning of next part
            flags=re.DOTALL | re.MULTILINE
        )
        parts = [{
            'part': int(match.group('part')),
            'solution': match.group('solution')
        } for match in part_regex.finditer(source)]
        # The last solution extends all the way to the validation code,
        # so we strip any trailing whitespace from it.
        parts[-1]['solution'] = parts[-1]['solution'].rstrip()
        return parts

    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = '{0}.{1}'.format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_parts(parts, url, token):
        submitted_parts = []
        for part in parts:
            if Check.has_solution(part):
                submitted_part = {
                    'part': part['part'],
                    'solution': part['solution'],
                    'valid': part['valid'],
                    'secret': [x for (x, _) in part['secret']],
                    'feedback': json.dumps(part['feedback']),
                }
                if 'token' in part:
                    submitted_part['token'] = part['token']
                submitted_parts.append(submitted_part)
        data = json.dumps(submitted_parts).encode('utf-8')
        headers = {
            'Authorization': token,
            'content-type': 'application/json'
        }
        request = urllib.request.Request(url, data=data, headers=headers)
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode('utf-8'))

    def update_attempts(old_parts, response):
        updates = {}
        for part in response['attempts']:
            part['feedback'] = json.loads(part['feedback'])
            updates[part['part']] = part
        for part in old_parts:
            valid_before = part['valid']
            part.update(updates.get(part['part'], {}))
            valid_after = part['valid']
            if valid_before and not valid_after:
                wrong_index = response['wrong_indices'].get(str(part['part']))
                if wrong_index is not None:
                    hint = part['secret'][wrong_index][1]
                    if hint:
                        part['feedback'].append('Namig: {}'.format(hint))


    filename = os.path.abspath(sys.argv[0])
    file_parts = extract_parts(filename)
    Check.initialize(file_parts)

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1Njc2fQ:1iIycp:PGXc8kq5dMGRvH_TOCPL_Q6KyAs'
        try:
            test_data = [
                ("""brez_ponovitev(['svinjski zrezek v omaki', 'sirov kanelon', 'ocvrt oslič', 'svinjski zrezek v omaki', 'ocvrt oslič', 'sirov burek', 'sirov kanelon', 'ocvrt oslič', 'sirov kanelon', 'sirov kanelon'])""", ['svinjski zrezek v omaki', 'sirov kanelon', 'ocvrt oslič', 'sirov burek']),
                ("""brez_ponovitev(['svinjski zrezek v omaki', 'zelenjavna rižota', 'puranji zrezek v gobovi omaki', 'sojini polpeti', 'sojini polpeti', 'puranji zrezek v gobovi omaki', 'zelenjavna rižota', 'svinjski zrezek v omaki'])""", ['svinjski zrezek v omaki', 'zelenjavna rižota', 'puranji zrezek v gobovi omaki', 'sojini polpeti']),
                ("""brez_ponovitev(['burek'])""", ['burek']),
                ("""brez_ponovitev(['burek', 'burek', 'burek', 'burek', 'burek', 'burek', 'burek', 'burek'])""", ['burek']),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1Njc3fQ:1iIycp:Uk9r3ZW2xOOQxsz46EgsqViy5Wo'
        try:
            test_data = [
                ("""kdaj_prej(['svinjski zrezek v omaki', 'sirov kanelon', 'ocvrt oslič', 'svinjski zrezek v omaki', 'ocvrt oslič', 'sirov burek', 'sirov kanelon', 'ocvrt oslič', 'sirov kanelon', 'sirov kanelon'])""", [7, 2, 5, 3, 2, 10, 5, 3, 2, 1]),
                ("""kdaj_prej(['svinjski zrezek v omaki', 'zelenjavna rižota', 'puranji zrezek v gobovi omaki', 'sojini polpeti', 'sojini polpeti', 'puranji zrezek v gobovi omaki', 'zelenjavna rižota', 'svinjski zrezek v omaki'])""", [1, 3, 5, 7, 1, 3, 5, 7]),
                ("""kdaj_prej(['burek'])""", [1]),
                ("""kdaj_prej(['burek', 'burek', 'burek', 'burek', 'burek', 'burek', 'burek', 'burek'])""", [1, 1, 1, 1, 1, 1, 1, 1]),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1Njc4fQ:1iIycp:gB3I1A9CVtMVUZ2-F6M7qVMqKcU'
        try:
            test_data = [
                ("""raznolikost_in_kakovost(['svinjski zrezek v omaki', 'sirov kanelon', 'ocvrt oslič', 'svinjski zrezek v omaki', 'ocvrt oslič', 'sirov burek', 'sirov kanelon', 'ocvrt oslič', 'sirov kanelon', 'sirov kanelon'])""", (4, 23.0)),
                ("""raznolikost_in_kakovost(['svinjski zrezek v omaki', 'zelenjavna rižota', 'puranji zrezek v gobovi omaki', 'sojini polpeti', 'sojini polpeti', 'puranji zrezek v gobovi omaki', 'zelenjavna rižota', 'svinjski zrezek v omaki'])""", (4, 21.0)),
                ("""raznolikost_in_kakovost(['burek'])""", (1, 1.0)),
                ("""raznolikost_in_kakovost(['burek', 'burek', 'burek', 'burek', 'burek', 'burek', 'burek', 'burek'])""", (1, 1.0)),
                ("""raznolikost_in_kakovost(['burek', 'burek', 'jogurt'])""", (2, 4.666666666666667)),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1Njc5fQ:1iIycp:muDQzOkd4O6OhGnrESOynmZmWAo'
        try:
            test_data = [
                ("""naj_jedilnik([['burek'], ['burek', 'burek', 'jogurt']])""", ['burek', 'burek', 'jogurt']),
                ("""naj_jedilnik([['burek', 'jogurt'], ['pomaranča', 'pomaranča']])""", ['burek', 'jogurt']),
                ("""naj_jedilnik([['svinjski zrezek v omaki', 'sirov kanelon', 'ocvrt oslič', 'svinjski zrezek v omaki', 'ocvrt oslič', 'sirov burek', 'sirov kanelon', 'ocvrt oslič', 'sirov kanelon', 'sirov kanelon'], ['svinjski zrezek v omaki', 'zelenjavna rižota', 'puranji zrezek v gobovi omaki', 'sojini polpeti', 'sojini polpeti', 'puranji zrezek v gobovi omaki', 'zelenjavna rižota', 'svinjski zrezek v omaki']])""",
                 ['svinjski zrezek v omaki', 'sirov kanelon', 'ocvrt oslič', 'svinjski zrezek v omaki', 'ocvrt oslič', 'sirov burek', 'sirov kanelon', 'ocvrt oslič', 'sirov kanelon', 'sirov kanelon']),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    print('Shranjujem rešitve na strežnik... ', end="")
    try:
        url = 'https://www.projekt-tomo.si/api/attempts/submit/'
        token = 'Token b8c7df44b9c20c8359c39be75a6d39215c0b0a7f'
        response = submit_parts(Check.parts, url, token)
    except urllib.error.URLError:
        print('PRI SHRANJEVANJU JE PRIŠLO DO NAPAKE! Poskusite znova.')
    else:
        print('Rešitve so shranjene.')
        update_attempts(Check.parts, response)
        if 'update' in response:
            print('Posodabljam datoteko... ', end="")
            backup_filename = backup(filename)
            with open(__file__, 'w', encoding='utf-8') as f:
                f.write(response['update'])
            print('Stara datoteka je bila preimenovana v {0}.'.format(backup_filename))
            print('Če se datoteka v urejevalniku ni osvežila, jo zaprite ter ponovno odprite.')
    Check.summarize()

if __name__ == '__main__':
    _validate_current_file()
