use std::collections::HashSet;

use crate::{parse_num, Part};
use itertools::Itertools;
use nalgebra::base::{Matrix3, Vector3};

struct Region {
    visible_beacons: HashSet<Vector3<i64>>,
    scanner_positions: Vec<Vector3<i64>>,
}

fn rotation_matrices() -> Vec<Matrix3<i64>> {
    // find all rotation matrices by first selecting one of the 6 directions for the x vector
    // and then picking one of the 4 directions that are orthogonal to it for the y vector
    // z = cross(x, y)
    let orientations = [
        Vector3::new(1, 0, 0i64),
        Vector3::new(-1, 0, 0),
        Vector3::new(0, 1, 0),
        Vector3::new(0, -1, 0),
        Vector3::new(0, 0, 1),
        Vector3::new(0, 0, -1),
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
fn find_merge_params(reg1: &Region, reg2: &Region) -> Option<(Matrix3<i64>, Vector3<i64>)> {
    for b1 in &reg1.visible_beacons {
        for b2 in &reg2.visible_beacons {
            for rotation in rotation_matrices() {
                let b2 = rotation * b2;
                let offset = b2 - b1;
                let potentially_common_beacons = reg2
                    .visible_beacons
                    .iter()
                    .map(|b| rotation * b - offset)
                    .filter(|b| reg1.visible_beacons.contains(b));
                if potentially_common_beacons.clone().count() >= 12 {
                    return Some((rotation, offset));
                }
            }
        }
    }
    None
}

// could avoid some cloning here
fn try_merge(reg1: &Region, reg2: &Region) -> Option<Region> {
    let (rotation, offset) = find_merge_params(reg1, reg2)?;
    let mut visible_beacons = reg1.visible_beacons.clone();
    visible_beacons.extend(reg2.visible_beacons.iter().map(|&b| rotation * b - offset));
    let mut scanner_positions = reg1.scanner_positions.clone();
    scanner_positions.extend(
        reg2.scanner_positions
            .iter()
            .map(|&pos| rotation * pos - offset),
    );
    Some(Region {
        visible_beacons,
        scanner_positions,
    })
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
                    Vector3::new(parse_num(x), parse_num(y), parse_num(z))
                })
                .collect();
            Region {
                visible_beacons,
                scanner_positions: vec![Vector3::new(0, 0, 0)],
            }
        })
        .collect_vec();

    'outer: while regions.len() > 1 {
        for reg1 in 0..regions.len() {
            for reg2 in reg1 + 1..regions.len() {
                if let Some(merged_region) = try_merge(&regions[reg1], &regions[reg2]) {
                    regions[reg1] = merged_region;
                    regions.swap_remove(reg2);
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
