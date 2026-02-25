from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

# 리터럴 : 양수는 변수, 음수는 변수의 부정을 의미
Lit = int
# Clause : 리터럴의 리스트로 OR로 연결된 형태 (절을 의미)
# 예: (x1 OR ¬x2 OR x3) 는 [1, -2, 3] 으로 표현
Clause = List[Lit]
# CNF : 절들의 리스트로 AND로 연결된 형태
# 예: (x1 OR ¬x2) AND (¬x1 OR x3) 는 [[1, -2], [-1, 3]] 으로 표현
CNF = List[Clause]


class Prop: ...

# @dataclass : 생성자, 보기 좋은 출력 문자열, 동등성 비교 등을 자동으로 생성해주는 데코레이터
# frozen=True : 인스턴스가 불변(immutable)임을 나타냄

@dataclass(frozen=True)
class TrueProp(Prop): ...

@dataclass(frozen=True)
class FalseProp(Prop): ...

@dataclass(frozen=True)
class VarProp(Prop):
    name: str

# c X x >= b
@dataclass(frozen=True)
class InequProp(Prop):
    c: float
    x: str
    b: float

# c1*x1 + c2*x2 + ... + cn*xn >= b (다중변수 선형 부등식)
@dataclass(frozen=True)
class LinearInequProp(Prop):
    coeffs: Dict[str, float]  # 변수명 -> 계수
    bound: float              # b (우변)

@dataclass(frozen=True)
class AndProp(Prop):
    p: Prop
    q: Prop

@dataclass(frozen=True)
class OrProp(Prop):
    p: Prop
    q: Prop

@dataclass(frozen=True)
class NotProp(Prop):
    p: Prop

# p → q
@dataclass(frozen=True)
class ImplProp(Prop):
    p: Prop
    q: Prop

# p ↔ q
@dataclass(frozen=True)
class IffProp(Prop):
    p: Prop
    q: Prop

# Example usage: expr = (p ∧ ¬q) → r
#  expr = ImplProp(
#     AndProp(VarProp("p"), NotProp(VarProp("q"))),
#     VarProp("r")
# )

# 새로운 변수 생성기
# 각 호출 시마다 고유한 정수 변수를 반환
class Fresh:
    def __init__(self) -> None:
        self.next_var = 1
    def new(self) -> int:
        v = self.next_var
        self.next_var += 1
        return v


# NNF (Negation Normal Form) 변환
# - →, ↔ 제거
# - ¬를 원자(VarProp, InequProp, TrueProp, FalseProp) 위로 밀어냄
# 결과적으로 NotProp은 원자 위에만 남게 됨
def to_nnf(prop: Prop) -> Prop:
    def elim(p: Prop) -> Prop:
        # →, ↔ 를 제거
        if isinstance(p, (VarProp, InequProp, TrueProp, FalseProp)):
            return p
        if isinstance(p, NotProp):
            return NotProp(elim(p.p))
        if isinstance(p, AndProp):
            return AndProp(elim(p.p), elim(p.q))
        if isinstance(p, OrProp):
            return OrProp(elim(p.p), elim(p.q))
        if isinstance(p, ImplProp):
            # p -> q == ¬p ∨ q
            return OrProp(NotProp(elim(p.p)), elim(p.q))
        if isinstance(p, IffProp):
            # p ↔ q == (p -> q) ∧ (q -> p)
            return AndProp(elim(ImplProp(p.p, p.q)), elim(ImplProp(p.q, p.p)))
        raise TypeError(f"Unknown Prop node in elim: {type(p).__name__}")

    def push(p: Prop, neg: bool) -> Prop:
        # neg=False: 정상, neg=True: 부정 상태(부정이 적용된 상황)
        if isinstance(p, TrueProp):
            return FalseProp() if neg else TrueProp()
        if isinstance(p, FalseProp):
            return TrueProp() if neg else FalseProp()
        if isinstance(p, VarProp) or isinstance(p, InequProp):
            return NotProp(p) if neg else p
        if isinstance(p, NotProp):
            # ¬(X) => 상태 반전
            return push(p.p, not neg)
        if isinstance(p, AndProp):
            if not neg:
                return AndProp(push(p.p, False), push(p.q, False))
            else:
                # ¬(A ∧ B) == ¬A ∨ ¬B
                return OrProp(push(p.p, True), push(p.q, True))
        if isinstance(p, OrProp):
            if not neg:
                return OrProp(push(p.p, False), push(p.q, False))
            else:
                # ¬(A ∨ B) == ¬A ∧ ¬B
                return AndProp(push(p.p, True), push(p.q, True))
        # Impl/Iff는 elim에서 제거되어야 함
        raise TypeError(f"Unknown Prop node in push: {type(p).__name__}")

    return push(elim(prop), False)


# Tseitin 변환 함수
# 입력: Prop 트리의 루트 노드
# Prop 트리는 부분식을 나타내는 노드들로 구성
# 출력: CNF 절 목록
def tseitin_to_cnf(root: Prop) -> Tuple[CNF, Dict[Prop, int]]:
    fresh = Fresh()
    cnf: CNF = []
    # Prop 노드에서 변수 ID로의 매핑
    # 예시 : node_to_var[NotProp(VarProp("q"))] = 7
    # 의미: SAT 변수 x7이 “(¬q) 부분식의 참/거짓”을 대표한다.
    # 장점 :
    # 같은 구조의 노드가 여러 번 나오면(구조적으로 동일한 dataclass 객체라면)
    # 같은 SAT 변수로 재사용되어 CNF가 불필요하게 커지는 것을 줄일 수 있습니다.
    node_to_var: Dict[Prop, int] = {}
    # atomic name -> var id (원자 단위는 이름 기반으로 변수를 재활용)
    atom_map: Dict[str, int] = {}

    # 먼저 NNF로 변환하여 ¬가 원자 위로만 오도록 정리
    root = to_nnf(root)

    # node가 나타내는 논리식을 대표하는 SAT 변수 번호 rep를 리턴
    # 동시에, rep ↔ node가 되도록 하는 CNF 절들을 cnf에 “추가”함
    def v(node: Prop) -> int:
        # 이미 처리된 노드라면, 매핑된 변수를 재사용
        # 절 추가 X (중복 방지)
        if node in node_to_var:
            return node_to_var[node]

        # 원자(변수, 부등식)는 이름/표현식 기반으로 고유 id를 할당하고
        # 새로운 fresh 변수를 직접 생성하지 않음(원자 자체가 변수이므로)
        if isinstance(node, VarProp):
            name = node.name
            if name not in atom_map:
                atom_map[name] = fresh.new()
            node_to_var[node] = atom_map[name]
            return node_to_var[node]

        if isinstance(node, InequProp):
            key = f"ineq:{node.c}:{node.x}:{node.b}"
            if key not in atom_map:
                atom_map[key] = fresh.new()
            node_to_var[node] = atom_map[key]
            return node_to_var[node]
        
        if isinstance(node, NotProp):
            inner = node.p

            # 케이스 1) ¬p (p가 원자) => 새 변수 만들지 말고 -id(p) 리턴
            if isinstance(inner, VarProp):
                return -v(inner)   # v(inner)는 atom_map을 통해 기존 id 재사용
            # 또한 부등식(InequProp) 같은 원자에 대한 부정도 처리
            if isinstance(inner, InequProp):
                return -v(inner)

        # 비원자 노드에 대해서만 새로운 변수를 할당
        rep = fresh.new()
        node_to_var[node] = rep

        # rep <-> node(논리식) 을 cnf에 추가함으로써
        # rep 이 해당 node와 같은 진리 값을 갖도록 제약 추가

        # node가 항상 True인 경우
        if isinstance(node, TrueProp):
            cnf.append([rep])      # rep is True
            return rep
        
        # node가 항상 False인 경우
        if isinstance(node, FalseProp):
            cnf.append([-rep])     # rep is False
            return rep

        # 논리 연산자 처리
        # 각 연산자에 대해 rep ↔ (연산자에 해당하는 식) 형태의 CNF 절 추가
        # <-> 는 두 방향의 함의를 모두 CNF로 변환하여 추가

       

        if isinstance(node, AndProp):
            a = v(node.p)
            b = v(node.q)
            # rep <-> (a ∧ b):
            # (¬a ∨ ¬b ∨ rep) ∧ (¬rep ∨ a) ∧ (¬rep ∨ b)
            cnf.append([-rep, a])
            cnf.append([-rep, b])
            cnf.append([ rep, -a, -b])
            return rep

        if isinstance(node, OrProp):
            a = v(node.p)
            b = v(node.q)
            # rep <-> (a ∨ b):
            # (a ∨ b ∨ ¬rep) ∧ (¬a ∨ rep) ∧ (¬b ∨ rep)
            cnf.append([-a, rep])
            cnf.append([-b, rep])
            cnf.append([-rep, a, b])
            return rep

        if isinstance(node, ImplProp):
            p = v(node.p)
            q = v(node.q)
            # (p -> q) == (¬p ∨ q)
            # rep <-> (¬p ∨ q):
            # (p ∨ rep) ∧ (¬q ∨ rep) ∧ (¬rep ∨ ¬p ∨ q)
            cnf.append([ p, rep])      
            cnf.append([-q, rep])       
            cnf.append([-rep, -p, q])   
            return rep
        
        if isinstance(node, IffProp):
            p = v(node.p)
            q = v(node.q)

            # a <-> (¬p ∨ q)   i.e., (p -> q)
            a = fresh.new()
            # (a -> (¬p ∨ q)) and ((¬p ∨ q) -> a)
            # OR 인코딩: a <-> (x ∨ y) gives (¬x ∨ a) ∧ (¬y ∨ a) ∧ (¬a ∨ x ∨ y)
            # 여기서 x=¬p, y=q
            cnf.append([ p,  a])        # ¬(¬p) ∨ a  == p ∨ a
            cnf.append([-q, a])         # ¬q ∨ a
            cnf.append([-a, -p, q])     # ¬a ∨ ¬p ∨ q

            # b <-> (¬q ∨ p)   i.e., (q -> p)
            b = fresh.new()
            cnf.append([ q,  b])        # q ∨ b   (¬(¬q) ∨ b)
            cnf.append([-p, b])         # ¬p ∨ b
            cnf.append([-b, -q, p])     # ¬b ∨ ¬q ∨ p

            # rep <-> (a ∧ b)
            # AND 인코딩: rep <-> (a ∧ b)
            cnf.append([-rep, a])       # ¬rep ∨ a
            cnf.append([-rep, b])       # ¬rep ∨ b
            cnf.append([ rep, -a, -b])  # rep ∨ ¬a ∨ ¬b

            return rep

        raise TypeError(f"Unknown Prop node: {type(node).__name__}")

    # 주어진 식의 루트 노드의 변수를 구하고,
    # 그 변수가 참이 되도록 하는 절을 추가해서 전체 식이 참이 되도록 강제
    # 이후 v함수 호출 시, 치환한 변수들이 해당하는 각 부분식의 진리 값을 나타내도록 보장
    root_var = v(root)
    cnf.append([root_var]) # 루트 노드가 참이도록 강제
    # 최종 CNF와 노드-변수 매핑 반환
    return cnf, node_to_var



# 출력함수


def _default_prop_to_str(prop: Prop) -> str:
    # VarProp(name="p") 같은 경우
    if hasattr(prop, "name"):
        return str(getattr(prop, "name"))

    # InequProp(c=..., x=..., b=...) 같은 경우
    if prop.__class__.__name__ == "InequProp":
        c = getattr(prop, "c", None)
        x = getattr(prop, "x", None)
        b = getattr(prop, "b", None)
        if c is not None and x is not None and b is not None:
            return f"({c}*{x} ≥ {b})"

    # 연산 노드들
    cls = prop.__class__.__name__
    if cls == "NotProp":
        return f"¬({_default_prop_to_str(getattr(prop, 'p'))})"
    if cls == "AndProp":
        return f"({_default_prop_to_str(getattr(prop, 'p'))} ∧ {_default_prop_to_str(getattr(prop, 'q'))})"
    if cls == "OrProp":
        return f"({_default_prop_to_str(getattr(prop, 'p'))} ∨ {_default_prop_to_str(getattr(prop, 'q'))})"
    if cls == "ImplProp":
        return f"({_default_prop_to_str(getattr(prop, 'p'))} → {_default_prop_to_str(getattr(prop, 'q'))})"
    if cls == "IffProp":
        return f"({_default_prop_to_str(getattr(prop, 'p'))} ↔ {_default_prop_to_str(getattr(prop, 'q'))})"

    # fallback
    return f"{cls}({prop})"


def _lit_to_str(lit: int, id_to_name: Dict[int, str]) -> str:
    v = abs(lit)
    name = id_to_name.get(v, f"v{v}")
    return f"¬{name}" if lit < 0 else name


def _clause_to_bool(clause: Clause, id_to_name: Dict[int, str]) -> str:
    if len(clause) == 0:
        return "FALSE"  # empty clause
    return "(" + " OR ".join(_lit_to_str(l, id_to_name) for l in clause) + ")"


def print_tseitin_result(
    cnf: CNF,
    node_to_var: Dict[Prop, int],
    *,
    root: Optional[Prop] = None,
    prop_to_str: Callable[[Prop], str] = _default_prop_to_str,
    sort_by_var: bool = True,
    show_only_top_k_clauses: Optional[int] = None,
) -> None:
    # 출력 모드 설정
    cnf_print_mode: str = "v_only"  # "v_only" | "named"

    # var_id -> "이름" 맵
    # 기본은 Prop 문자열을 붙이는데, 같은 var_id에 여러 Prop가 매핑되면(보통은 없음)
    # 가장 짧은 표현을 택함.
    id_to_name: Dict[int, str] = {}
    for prop, vid in node_to_var.items():
        s = prop_to_str(prop)
        if vid not in id_to_name or len(s) < len(id_to_name[vid]):
            id_to_name[vid] = s

    def _cnf_to_bool_v_only(cnf: list[list[int]]) -> str:
        def lit_to_str(lit: int) -> str:
            vid = abs(lit)
            return f"v{vid}" if lit > 0 else f"¬v{vid}"

        def clause_to_str(clause: list[int]) -> str:
            return "(" + " OR ".join(lit_to_str(l) for l in clause) + ")"

        if not cnf:
            return "TRUE"
        return " AND ".join(clause_to_str(cl) for cl in cnf)

    def _negate_str(atom: str) -> str:
        a = atom.strip()

        # "¬(X)"면 바깥 ¬를 제거해서 X 반환 (¬¬X 단순화)
        if a.startswith("¬(") and a.endswith(")"):
            return a[2:-1].strip()

        # "¬X"면 ¬만 제거
        if a.startswith("¬"):
            return a[1:].strip()

        # 일반 부정
        if a.startswith("(") and a.endswith(")"):
            return "¬" + a
        return f"¬({a})"


    def _cnf_to_bool_named(cnf: list[list[int]], id_to_name: dict[int, str]) -> str:
        def lit_to_str(lit: int) -> str:
            vid = abs(lit)
            base = id_to_name.get(vid, f"v{vid}")
            return base if lit > 0 else _negate_str(base)

        def clause_to_str(clause: list[int]) -> str:
            return "(" + " OR ".join(lit_to_str(l) for l in clause) + ")"

        if not cnf:
            return "TRUE"
        return " AND ".join(clause_to_str(cl) for cl in cnf)

    # CNF를 불리언 함수로 출력
    if show_only_top_k_clauses is None:
        cnf_for_print = cnf
    else:
        cnf_for_print = cnf[:show_only_top_k_clauses]

    print("=== Tseitin CNF (boolean function form) ===")
    if cnf_print_mode == "v_only":
        print(_cnf_to_bool_v_only(cnf_for_print))
    else:
        print(_cnf_to_bool_named(cnf_for_print, id_to_name))

    # 루트 강조(있으면)
    if root is not None:
        print("\n=== Root mapping ===")
        # 원본 루트 정보
        rv = node_to_var.get(root)
        if rv is None:
            print("before NNF root:   ", prop_to_str(root))
        else:
            print(f"root: v{rv}  <->  {prop_to_str(root)}")
            print(f"root unit clause (should exist): (v{rv})")

        # NNF로 변환한 형태도 함께 보여주기
        try:
            nnf_root = to_nnf(root)
            print("\n=== Root in NNF ===")
            print(f"NNF : {prop_to_str(nnf_root)}")
            nnf_rv = node_to_var.get(nnf_root)
            if nnf_rv is None:
                print("nnf root: (not found in node_to_var)  ", prop_to_str(nnf_root))
            else:
                print(f"nnf root: v{nnf_rv}  <->  {prop_to_str(nnf_root)}")
                print(f"nnf root unit clause (if present): (v{nnf_rv})")
        except Exception as e:
            print(f"(Failed to compute NNF for root: {e})")

    # 부분식 -> 변수 매핑 출력
    items = list(node_to_var.items())

    def key_fn(kv):
        prop, vid = kv
        return (vid, prop_to_str(prop)) if sort_by_var else (prop_to_str(prop), vid)

    items.sort(key=key_fn)

    print("\n=== Subformula -> Var mapping ===")
    for prop, vid in items:
        print(f"v{vid:4d}  <->  {prop_to_str(prop)}")

    # 원하면 원래 DIMACS 절도 같이
    print("\n=== Raw CNF clauses (DIMACS ints) ===")
    for i, clause in enumerate(cnf_for_print, 1):
        print(f"{i:4d}: {clause}")


def main() -> None:

    p = VarProp("p")
    q = VarProp("q")

    phi = NotProp( AndProp( VarProp( "p" ), VarProp( "q" ) ) )

    # --- Tseitin 변환 실행 ---
    cnf, node_to_var = tseitin_to_cnf(phi)

    # --- 결과 출력(앞서 만든 프린트 함수 사용) ---
    print_tseitin_result(
        cnf,
        node_to_var,
        root=phi,
        show_only_top_k_clauses=200,  # 너무 길면 줄이기
    )


if __name__ == "__main__":
    main()