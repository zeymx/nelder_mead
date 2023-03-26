import nelder_mead


def main() -> None:
    nm = nelder_mead.NelderMead()
    pbest = []
    pbest, value = nm.execute_nelder_mead(4)
    print(f"Minimum of given function is {value}, reached in point {pbest}")

if __name__ == "__main__":
    main()
