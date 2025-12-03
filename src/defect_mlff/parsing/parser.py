from pathlib import Path
from typing import List, Iterable, Iterator
from ase.atoms import Atoms
from ase.io import read, write

class AimsPAXparser:
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.ase_list: List[Atoms] = self.parse_xyz(self.input_path)

    def parse_xyz(self, xyz_path: Path) -> List[Atoms]:
        # Load ALL frames from the xyz (works for single- or multi-frame files)
        frames = read(xyz_path, index=':')
        # ASE returns a list for multi-frame; ensure list in all cases
        if isinstance(frames, Atoms):
            return [frames]
        return list(frames)

    # Convenience methods
    def __len__(self) -> int:
        return len(self.ase_list)

    def __getitem__(self, idx) -> Atoms:
        return self.ase_list[idx]

    def __iter__(self) -> Iterator[Atoms]:
        return iter(self.ase_list)

    def write_geometries(self, out_dir: str = "geometries", pattern: str = "geometry.{i:03d}.in") -> List[Path]:
        """
        Write each frame to its own FHI-aims geometry file.
        Example outputs: geometry.000.in, geometry.001.in, ...
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        written = []
        for i, atoms in enumerate(self.ase_list):
            fname = out_path / pattern.format(i=i)
            write(fname, atoms, format="aims")  # FHI-aims writer
            written.append(fname)
        return written

if __name__ == "__main__":
    # Point this at your file
    parser = AimsPAXparser("files/aims-pax_example.xyz")
    print(f"Loaded {len(parser)} frame(s).")
    # Example: access frames
    first = parser[0]
    # Optional: write each frame to its own geometry file
    out_paths = parser.write_geometries()
    print("Wrote:")
    for p in out_paths:
        print(" -", p)