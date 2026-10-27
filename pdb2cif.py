#!/usr/bin/env python3
"""Convert a PDB structure to mmCIF with consistent molecular-entity metadata.

Preserves all atoms, models, chains, ligands and waters represented by Gemmi.
Normalizes AMBER/GROMACS atom names ILE CD -> CD1 and terminal OC1/OC2 -> O/OXT.
For PDBs without SEQRES records, polymer sequences are inferred from observed
residues (which may omit unresolved residues). Does not reconstruct atoms.

Usage: python pdb2cif.py -i input.pdb -o output.cif
Requires: pip install gemmi
"""

import argparse
import sys
from pathlib import Path

import gemmi


def observed_sequence(structure, entity):
    """Return residues from the first available polymer copy of an entity."""
    for model in structure:
        for chain in model:
            residues = [
                residue.name for residue in chain
                if residue.subchain in entity.subchains
                and residue.entity_type == gemmi.EntityType.Polymer
            ]
            if residues:
                return residues
    return []


def canonical_one_letter(sequence, polymer_type):
    """Use Gemmi's residue table, with an explicit unknown fallback."""
    unknown = 'N' if polymer_type in (
        gemmi.PolymerType.Rna,
        gemmi.PolymerType.Dna,
        gemmi.PolymerType.DnaRnaHybrid,
    ) else 'X'
    chars = []
    for code in sequence:
        info = gemmi.find_tabulated_residue(code)
        letter = info.one_letter_code.upper()
        chars.append(letter if len(letter) == 1 and letter.isalpha() else unknown)
    return ''.join(chars)



def assign_missing_chain_ids(structure):
    """Give unnamed PDB chains stable identifiers so Gemmi can create entities.

    Some MD-generated PDBs leave column 22 (chain ID) blank. Gemmi parses
    their atoms correctly but can fail to create entities from unnamed chains.
    Only missing chain names are assigned; already named chains are untouched.
    The same nth unnamed chain gets the same name in every coordinate model.
    """
    reserved = {
        chain.name
        for model in structure
        for chain in model
        if chain.name.strip()
    }
    available = (
        ch for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        if ch not in reserved
    )
    assigned = []
    for model in structure:
        unnamed_idx = 0
        for chain in model:
            if chain.name.strip():
                continue
            if unnamed_idx == len(assigned):
                try:
                    assigned.append(next(available))
                except StopIteration:
                    raise ValueError('Too many unnamed chains to assign PDB chain IDs.')
            chain.name = assigned[unnamed_idx]
            unnamed_idx += 1
    return assigned


def normalize_isoleucine_atom_names(structure):
    """Rename AMBER/GROMACS ILE CD to the standard PDB atom name CD1.

    Only atom names change; no coordinates or other residues are modified.
    Refuse ambiguous residues that already contain both names.
    """
    renamed = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name != 'ILE':
                    continue
                atom_names = {atom.name for atom in residue}
                if 'CD' not in atom_names:
                    continue
                if 'CD1' in atom_names:
                    raise ValueError(
                        f'ILE residue {residue.seqid} in chain '
                        f'{chain.name!r} contains both CD and CD1; '
                        'cannot rename unambiguously.'
                    )
                for atom in residue:
                    if atom.name == 'CD':
                        atom.name = 'CD1'
                        renamed += 1
    return renamed



def normalize_c_terminal_oxygen_names(structure):
    """Convert GROMACS terminal OC1/OC2 to standard PDB O/OXT.

    Only the final amino-acid residue of each chain is eligible. All atom
    coordinates are preserved. Refuse ambiguous pre-existing oxygen names.
    """
    renamed = 0
    for model in structure:
        for chain in model:
            residues = list(chain)
            amino_indices = [
                i for i, residue in enumerate(residues)
                if gemmi.find_tabulated_residue(residue.name).is_amino_acid()
            ]
            if not amino_indices:
                continue
            terminal_idx = amino_indices[-1]
            for i in amino_indices:
                residue = residues[i]
                names = {atom.name for atom in residue}
                if not (names & {'OC1', 'OC2'}):
                    continue
                if i != terminal_idx:
                    raise ValueError(
                        f'Unexpected OC1/OC2 in internal residue '
                        f'{residue.name} {residue.seqid} of chain {chain.name!r}; '
                        'cannot normalize safely.'
                    )
                if 'OC1' not in names or 'OC2' not in names:
                    raise ValueError(
                        f'Terminal residue {residue.name} {residue.seqid} '
                        f'of chain {chain.name!r} has incomplete OC1/OC2 pair; '
                        'please inspect the terminal oxygen atoms.'
                    )
                if 'O' in names or 'OXT' in names:
                    raise ValueError(
                        f'Terminal residue {residue.name} {residue.seqid} '
                        f'of chain {chain.name!r} mixes OC1/OC2 with O/OXT; '
                        'cannot rename without duplicate names.'
                    )
                for atom in residue:
                    if atom.name == 'OC1':
                        atom.name = 'O'
                        renamed += 1
                    elif atom.name == 'OC2':
                        atom.name = 'OXT'
                        renamed += 1
    return renamed


def convert(input_path: Path, output_path: Path):
    if not input_path.is_file():
        raise FileNotFoundError(f'Input does not exist: {input_path}')

    structure = gemmi.read_structure(str(input_path))
    if len(structure) == 0:
        raise ValueError('Input contains no coordinate models.')

    # PDB files exported by MD tools commonly leave chain IDs blank.
    # Gemmi can read their atoms but may generate no entity records at all.
    # Assign only missing chain names before invoking Gemmi's entity logic.
    renamed_chains = assign_missing_chain_ids(structure)
    if renamed_chains:
        print(
            'WARNING: Blank PDB chain IDs were assigned: '
            + ', '.join(renamed_chains)
            + '. Atomic coordinates were not changed.',
            file=sys.stderr,
        )

    # Normalize a known force-field/PDB naming difference before writing CIF.
    renamed_ile_atoms = normalize_isoleucine_atom_names(structure)
    if renamed_ile_atoms:
        print(
            f'Normalized {renamed_ile_atoms} ILE atom name(s): CD -> CD1. '
            'Coordinates were not changed.'
        )

    # Convert GROMACS/AMBER C-terminal oxygen names to wwPDB O and OXT.
    renamed_terminal_atoms = normalize_c_terminal_oxygen_names(structure)
    if renamed_terminal_atoms:
        print(
            f'Normalized {renamed_terminal_atoms} terminal oxygen atom name(s): '
            'OC1 -> O, OC2 -> OXT. Coordinates were not changed.'
        )

    # Use Gemmi's heuristics to identify polymer/non-polymer/water entities.
    # No atoms are intentionally removed.
    structure.setup_entities()

    if not structure.entities:
        raise ValueError('Gemmi could not identify any molecular entities.')

    # Gemmi may use e.g. 'A', 'ATP!', 'water' as entity IDs when importing PDB.
    # Numeric IDs are standard PDBx/mmCIF and required by BioEmu2's importer.
    missing_sequence = []
    for index, entity in enumerate(structure.entities, start=1):
        entity.name = str(index)
        if entity.entity_type == gemmi.EntityType.Polymer and not entity.full_sequence:
            seq = observed_sequence(structure, entity)
            if not seq:
                raise ValueError(f'Cannot infer sequence for polymer entity {index}.')
            entity.full_sequence = seq
            missing_sequence.append(str(index))

    # Matches residue numbering to the full sequence when it is known.
    structure.assign_label_seq_id()
    block = structure.make_mmcif_document().sole_block()

    # Gemmi writes the unexpanded sequence, but can omit this canonical field.
    # Add it using the actual full sequence (SEQRES if available, otherwise
    # observed residues). This is standard PDBx/mmCIF entity metadata.
    poly_entities = {
        entity.name: entity
        for entity in structure.entities
        if entity.entity_type == gemmi.EntityType.Polymer
    }
    if poly_entities:
        item = block.find_loop_item('_entity_poly.entity_id')
        if item is None or item.loop is None:
            raise ValueError('Gemmi did not emit the _entity_poly metadata.')
        loop = item.loop
        tag = '_entity_poly.pdbx_seq_one_letter_code_can'
        if tag not in loop.tags:
            loop.add_columns([tag], '?')
        table = block.find(['_entity_poly.entity_id', tag])
        for row in table:
            entity = poly_entities.get(row[0])
            if entity is None:
                raise ValueError(f'Unknown polymer entity ID {row[0]}.')
            row[1] = canonical_one_letter(entity.full_sequence, entity.polymer_type)

    # Gemmi's writer may omit _pdbx_entity_nonpoly even though non-polymers
    # appear in _entity and _atom_site. Emit the standard entity -> CCD ID
    # mapping without discarding any ligands/ions/waters.
    comp_by_entity = {}
    for entity in structure.entities:
        if entity.entity_type not in (gemmi.EntityType.NonPolymer, gemmi.EntityType.Water):
            continue
        names = {
            residue.name
            for model in structure
            for chain in model
            for residue in chain
            if residue.subchain in entity.subchains
        }
        if len(names) != 1:
            raise ValueError(
                f'Cannot assign a unique component to entity {entity.name}: '
                f'{sorted(names)}'
            )
        comp_by_entity[entity.name] = next(iter(names))

    if comp_by_entity:
        item = block.find_loop_item('_pdbx_entity_nonpoly.entity_id')
        if item is None:
            loop = block.init_loop(
                '_pdbx_entity_nonpoly.', ['entity_id', 'comp_id']
            )
            for entity_id, comp_id in comp_by_entity.items():
                loop.add_row([entity_id, comp_id])
        else:
            # Unusual for Gemmi output, but check any preexisting mapping.
            rows = block.find([
                '_pdbx_entity_nonpoly.entity_id',
                '_pdbx_entity_nonpoly.comp_id',
            ])
            existing = {r[0]: r[1] for r in rows}
            if any(existing.get(k) != v for k, v in comp_by_entity.items()):
                raise ValueError('Existing non-polymer metadata is inconsistent.')

    # Verify the specific malformed entity-ID condition that previously
    # caused BioEmu2's int('.') exception.
    ids = list(block.find_values('_atom_site.label_entity_id'))
    if not ids or any(not value.isdecimal() for value in ids):
        bad = sorted({x for x in ids if not x.isdecimal()})
        raise ValueError(f'Invalid _atom_site.label_entity_id values: {bad}')

    # Check that Gemmi did not lose atoms in conversion.
    expected = sum(len(residue) for model in structure for chain in model for residue in chain)
    if len(ids) != expected:
        raise ValueError(f'Atom count mismatch: input {expected}, output {len(ids)}.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    block.write_file(str(output_path))
    print(f'Converted: {input_path} -> {output_path}')
    print(f'Atoms: {len(ids)}; models: {len(structure)}; entities: {len(structure.entities)}')
    if missing_sequence:
        print(
            'WARNING: Input lacked SEQRES/full sequence for polymer entity IDs '
            + ', '.join(missing_sequence)
            + '. Sequence was inferred from observed residues; unresolved '
            'residues cannot be recovered from coordinates.',
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDB to mmCIF, preserving molecules and fixing entity metadata.'
    )
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input PDB file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output mmCIF file')
    args = parser.parse_args()
    try:
        convert(args.input, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
