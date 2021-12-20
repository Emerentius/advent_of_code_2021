use std::collections::HashSet;

use crate::{parse_num, Part};
use itertools::Itertools;
use nalgebra::base::{Matrix3, Vector3};
use once_cell::sync::OnceCell;

// point or difference
type Vec3 = Vector3<i64>;
type RotMatrix = Matrix3<i64>;

struct Region {
    visible_beacons: HashSet<Vec3>,
    scanner_positions: Vec<Vec3>,
}

static ALL_ROTATIONS: OnceCell<Vec<RotMatrix>> = OnceCell::new();

fn rotation_matrices() -> Vec<RotMatrix> {
    // find all rotation matrices by first selecting one of the 6 directions for the x vector
    // and then picking one of the 4 directions that are orthogonal to it for the y vector
    // z = cross(x, y)
    let orientations = [
        Vec3::new(1, 0, 0i64),
        Vec3::new(-1, 0, 0),
        Vec3::new(0, 1, 0),
        Vec3::new(0, -1, 0),
        Vec3::new(0, 0, 1),
        Vec3::new(0, 0, -1),
    ];
    orientations
        .into_iter()
        .flat_map(|v1| {
            orientations
                .into_iter()
                .filter(move |v2| v1.dot(v2) == 0)
                .map(move |v2| (v1, v2))
        })
        .map(|(v1, v2)| {
            let v3 = v1.cross(&v2);
            #[rustfmt::skip]
            Matrix3::new(
                v1[0], v2[0], v3[0],
                v1[1], v2[1], v3[1],
                v1[2], v2[2], v3[2],
            )
        })
        .collect()
}

// Returns rotation and offset that moves reg2 onto reg1.
fn find_merge_params(reg1: &Region, reg2: &Region) -> Option<(RotMatrix, Vec3)> {
    for b1 in &reg1.visible_beacons {
        for b2 in &reg2.visible_beacons {
            for rotation in ALL_ROTATIONS.get().unwrap() {
                let b2 = rotation * b2;
                let offset = b2 - b1;
                let potentially_common_beacons = reg2
                    .visible_beacons
                    .iter()
                    .map(|b| rotation * b - offset)
                    .filter(|b| reg1.visible_beacons.contains(b));
                if potentially_common_beacons.clone().count() >= 12 {
                    return Some((*rotation, offset));
                }
            }
        }
    }
    None
}

// could avoid some cloning here
fn merge(mut reg1: Region, reg2: Region, rotation: RotMatrix, offset: Vec3) -> Region {
    let new_pos = |pos| rotation * pos - offset;
    reg1.visible_beacons
        .extend(reg2.visible_beacons.into_iter().map(new_pos));
    reg1.scanner_positions
        .extend(reg2.scanner_positions.into_iter().map(new_pos));
    reg1
}

pub fn day19(part: Part) {
    let input = include_str!("day19_input.txt");
    let mut regions = input
        .split("\n\n")
        .map(|scanner_data| {
            let visible_beacons = scanner_data
                .lines()
                .skip(1)
                .map(|beacon| {
                    let (_, x, y, z) =
                        lazy_regex::regex_captures!(r"(\-?\d+),(\-?\d+),(\-?\d+)", beacon).unwrap();
                    Vec3::new(parse_num(x), parse_num(y), parse_num(z))
                })
                .collect();
            Region {
                visible_beacons,
                scanner_positions: vec![Vec3::new(0, 0, 0)],
            }
        })
        .collect_vec();

    ALL_ROTATIONS.set(rotation_matrices()).unwrap();

    'outer: while regions.len() > 1 {
        for reg1 in 0..regions.len() {
            for reg2 in reg1 + 1..regions.len() {
                if let Some((rotation, offset)) = find_merge_params(&regions[reg1], &regions[reg2])
                {
                    let region2 = regions.swap_remove(reg2);
                    let region1 = regions.swap_remove(reg1);
                    let new_region = merge(region1, region2, rotation, offset);
                    regions.push(new_region);
                    continue 'outer;
                }
            }
        }
        unreachable!();
    }

    let region = regions.into_iter().next().unwrap();
    match part {
        Part::One => println!("{}", region.visible_beacons.len()),
        Part::Two => {
            let scanner_pairs =
                region
                    .scanner_positions
                    .iter()
                    .enumerate()
                    .flat_map(|(i, pos1)| {
                        region.scanner_positions[i + 1..]
                            .iter()
                            .map(move |pos2| (pos1, pos2))
                    });
            let distances = scanner_pairs
                .map(|(p1, p2)| (0..3).map(|dim| (p1[dim] - p2[dim]).abs()).sum::<i64>());
            let max_scanner_distance = distances.max().unwrap();
            println!("{}", max_scanner_distance);
        }
    }
}
