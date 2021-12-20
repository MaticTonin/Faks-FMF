# =============================================================================
# Disemvoweling
#
# Matej je moderator na nekem manj znanem forumu, ki ima kar lepo število
# registriranih uporabnikov, vendar večina med njimi piše same neumnosti.
# Med takimi je še posebno dejaven Tomaž Majer. Če se zdi Mateju objava
# žaljiva, jo nemudoma zbriše. Če objava na forumu ni žaljiva, ampak je le
# precej neumna pa naredi naslednje: Vse samoglasnike odstrani iz besedila
# in jih (v enakem vrstnem redu) doda nazaj na konec. Na primer sporočilo:
# 
#     Banana je najboljša! Izjemno dobra je tudi v kombinaciji z Nutello!
# 
# postane
# 
#     Bnn j njbljš! zjmn dbr j td v kmbncj z Ntll!aaaeaoaIeooaeuioiaiiueo
# 
# Takó besedilo postane težko berljivo in večina bralcev ga ignorira. Če
# pa se najde kdo, ki ga vsebina resnično zanima, lahko z nekaj truda
# razvozla vsebino sporočila.
# 
# (Opisana transformacija besedila je znana pod imenom disemvoweling.)
# =====================================================================@005658=
# 1. podnaloga
# Napišite funkcijo `prvi_samoglasnik(niz)`, ki kot argument dobi niz `niz`
# in vrne položaj prvega samoglasnika v danem nizu. Predpostavite lahko,
# da bo niz vseboval vsaj en samoglasnik. Zgled:
# 
#     >>> prvi_samoglasnik('Krščen matiček!')
#     4
# =============================================================================

def prvi_samoglasnik(niz):
    mali_samoglasniki= "aeiou"
    veliki_samoglasniki = mali_samoglasniki.upper()
    samoglasniki = veliki_samoglasniki + mali_samoglasniki
    for i, c in enumerate(niz):
        if c in samoglasniki:
            return i

# =====================================================================@005659=
# 2. podnaloga
# Napišite funkcijo `disemvowel(sporocilo)`, ki kot argument prejme niz
# `sporocilo`. Funkcija naj sestavi in vrne nov niz, ki ga dobi iz niza
# `sporocilo` tako, da vse samoglasnike prestavi na konec niza. Zgled:
# 
#     >>> disemvowel('Banana je dobra!')
#     'Bnn j dbr!aaaeoa'
# =============================================================================
def disemvowel(sporocilo):
    mali_samoglasniki="aeiou"
    veliki_samoglasniki = mali_samoglasniki.upper()
    samoglasniki = veliki_samoglasniki + mali_samoglasniki
    samo=""
    brez=""
    for i  in sporocilo:
        if i in samoglasniki:
            samo += i
        else: 
            brez+= i
    return brez + samo



# =====================================================================@005660=
# 3. podnaloga
# Sestavite še funkcijo `disemvowel_datoteko(vhodna, izhodna)`, ki kot
# prvi argument dobi _ime_ vhodne datoteke, kot drugi argument pa _ime_
# izhodne datoteke. Funkcija naj vsebino vhodne datoteke prepiše na izhodno
# datoteko, pri čemer naj na vsaki vrstici posebej izvede disemvoweling.
# 
# Če je na primer v vhodni datoteki besedilo
# 
#     Pred njo s kopiti čevljarček se ustavi;
#     ker ogleduje smolec obuvalo,
#     jermenov, meni, da ima premalo;
#     kar on očita, koj Apel popravi.
# 
# potem funkcija v izhodno datoteko zapiše:
# 
#     Prd nj s kpt čvljrčk s stv;eooiieaeeuai
#     kr gldj smlc bvl,eoeueoeouao
#     jrmnv, mn, d m prml;eeoeiaiaeao
#     kr n čt, kj pl pprv.aooiaoAeoai
# =============================================================================
def disemvowel_datoteko(vhodna, izhodna):
    with open(izhodna,"w") as f:
        for vrstica in open(vhodna):
            print(disemvowel(vrstica.strip()), file=f) 


# =====================================================================@005661=
# 4. podnaloga
# Včasih pa želimo sporočilo dešifrirati, saj nas zanima njegova vsebina.
# Ta naloga je vse prej kot enostavna, saj ne vemo kam točno je treba vriniti
# izgnane samoglasnike. Sporočilo `'Bnn j dbr!aaaeoa'` bi lahko dešifrirali
# tudi kot `'Banna ja debora!'`, kot `'Bnana aje dobar!'` ipd.
# 
# Matej je na mesta v besedilu, kjer predvideva, da bi morali stati
# samoglasniki, vstavil znake `'*'` (zvezdica). V sporočilu zvezdica nikoli
# ne bo imela drugega pomena. Prav tako bo število samoglasnikov enako številu
# zvezdic in vsi samoglasniki bodo na koncu niza.
# 
# Napišite funkcijo `razveljavi_disemvowel(niz)`, ki bo zvezdice v nizu
# nadomestila s samoglasniki, ki se v nizu `niz` nahajajo na koncu. Zgled:
# 
#     >>> razveljavi_disemvowel('B*n*n* j* d*br*!aaaeoa')
#     'Banana je dobra!'
# =============================================================================

def razveljavi_disemvowel(niz):
    st_zvezdic=niz.count("*")
    if st_zvezdic==0:
        return niz
    samoglasniki_v_nizu=niz[-st_zvezdic:]
    nov_niz=""
    stevec=0
    for i in niz[:-st_zvezdic]:
        if i == "*":
            nov_niz += samoglasniki_v_nizu[stevec]
            stevec+=1
        else: 
            nov_niz += i
    return nov_niz

    



































































































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
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NjU4fQ:1iHx9V:oNvwFeC07sfknb-ui-Aj_d1tL3E'
        try:
            test_data = [
                ("prvi_samoglasnik('Krščen matiček!')", 4),
                ("prvi_samoglasnik('Danes je lep dan!')", 1),
                ("prvi_samoglasnik('Prvi, drugi in tretji.')", 3),
                ("prvi_samoglasnik('Apel in čevljar')", 0),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NjU5fQ:1iHx9V:UvyttiYE3A94KYg_6N3m4qcPmnM'
        try:
            test_data = [
                ("disemvowel('Banana je dobra!')", 'Bnn j dbr!aaaeoa'),
                ("disemvowel('Pred njo s kopiti čevljarček se ustavi;')", 'Prd nj s kpt čvljrčk s stv;eooiieaeeuai'),
                ("disemvowel('puškomitraljez')", 'pškmtrljzuoiae'),
                ("disemvowel('Bnn j dbr!aaaeoa')", 'Bnn j dbr!aaaeoa'),
                ("disemvowel('3.141592653589793')", '3.141592653589793'),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NjYwfQ:1iHx9V:755UsXZ6JpCvKscOh_1W5XhMAIo'
        try:
            with Check.in_file("disemvoweling_vhod.txt", [
                "Pred njo s kopiti čevljarček se ustavi;",
                "ker ogleduje smolec obuvalo,",
                "jermenov, meni, da ima premalo;",
                "kar on očita, koj Apel popravi."
            ], encoding='utf-8'):
                disemvowel_datoteko("disemvoweling_vhod.txt", "disemvoweling_izhod.txt")
                Check.out_file("disemvoweling_izhod.txt", [
                    "Prd nj s kpt čvljrčk s stv;eooiieaeeuai",
                    "kr gldj smlc bvl,eoeueoeouao",
                    "jrmnv, mn, d m prml;eeoeiaiaeao",
                    "kr n čt, kj pl pprv.aooiaoAeoai"
                ], encoding='utf-8')
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NjYxfQ:1iHx9V:UMP_V8R7Ikhxxds735_xgEipg9E'
        try:
            test_data = [
                ("razveljavi_disemvowel('B*n*n* j* d*br*!aaaeoa')", 'Banana je dobra!'),
                ("razveljavi_disemvowel('Pr*d nj* s k*p*t* č*vlj*rč*k s* *st*v*;eooiieaeeuai')", 'Pred njo s kopiti čevljarček se ustavi;'),
                ("razveljavi_disemvowel('p*šk*m*tr*lj*zuoiae')", 'puškomitraljez'),
                ("razveljavi_disemvowel('Bnn j dbr!******aaaeoa')", 'Bnn j dbr!aaaeoa'),
                ("razveljavi_disemvowel('3.141592653589793')", '3.141592653589793'),
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
            print('Updating file... ', end="")
            backup_filename = backup(filename)
            with open(__file__, 'w', encoding='utf-8') as f:
                f.write(response['update'])
            print('Previous file has been renamed to {0}.'.format(backup_filename))
            print('If the file did not refresh in your editor, close and reopen it.')
    Check.summarize()

if __name__ == '__main__':
    _validate_current_file()
