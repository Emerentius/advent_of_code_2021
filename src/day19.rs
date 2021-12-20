use fnv::FnvHashSet as HashSet;

use crate::{parse_num, Part};
use itertools::Itertools;
use nalgebra::base::{Matrix3, Vector3};
use once_cell::sync::Lazy;

type Num = i16;
// point or difference
type Vec3 = Vector3<Num>;
type RotMatrix = Matrix3<Num>;

struct Region {
    beacons: HashSet<Vec3>,
    scanner: Vec<Vec3>,
}

static ALL_ROTATIONS: Lazy<Vec<RotMatrix>> = Lazy::new(|| {
    // find all rotation matrices by first selecting one of the 6 directions for the x vector
    // and then picking one of the 4 directions that are orthogonal to it for the y vector
    // z = cross(x, y)
    let axis_vecs = [Vec3::x(), Vec3::y(), Vec3::z()];
    let orientations = axis_vecs.into_iter().flat_map(|dir| [dir, -dir]);
    itertools::iproduct!(orientations.clone(), orientations)
        .filter(|(v1, v2)| v1.dot(v2) == 0)
        .map(|(v1, v2)| {
            let v3: Vec3 = v1.cross(&v2);
            let col_iter = itertools::chain!(&v1, &v2, &v3).copied();
            Matrix3::from_iterator(col_iter)
        })
        .collect()
});

// Returns rotation and offset that moves reg2 onto reg1.
fn find_merge_params(reg1: &Region, reg2: &Region) -> Option<(RotMatrix, Vec3)> {
    for rotation in &*ALL_ROTATIONS {
        let rotated_beacons2 = reg2.beacons.iter().map(|b| rotation * b).collect_vec();
        for b2 in &rotated_beacons2 {
            for b1 in &reg1.beacons {
                let offset = b2 - b1;
                let potentially_common_beacons = rotated_beacons2
                    .iter()
                    .map(|&b| b - offset)
                    .filter(|b| reg1.beacons.contains(b));
                if potentially_common_beacons.count() >= 12 {
                    return Some((*rotation, offset));
                }
            }
        }
    }
    None
}

fn merge(mut reg1: Region, reg2: Region, rotation: RotMatrix, offset: Vec3) -> Region {
    let new_pos = |pos| rotation * pos - offset;
    reg1.beacons.extend(reg2.beacons.into_iter().map(new_pos));
    reg1.scanner.extend(reg2.scanner.into_iter().map(new_pos));
    reg1
}

pub fn day19(part: Part) {
    let input = include_str!("day19_input.txt");
    let mut regions = input
        .split("\n\n")
        .map(|scanner_data| {
            let beacons = scanner_data
                .lines()
                .skip(1)
                .map(|beacon| Vec3::from_iterator(beacon.split(',').map(|num| parse_num(num) as _)))
                .collect();
            let scanner = vec![Vec3::zeros()];
            Region { beacons, scanner }
        })
        .collect_vec();

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
        Part::One => println!("{}", region.beacons.len()),
        Part::Two => {
            let manhattan_dist = |diff: Vec3| diff.iter().map(|&num| num.abs()).sum::<Num>();
            let scanner_pairs = region.scanner.iter().tuple_combinations();
            let distances = scanner_pairs.map(|(p1, p2)| p1 - p2).map(manhattan_dist);
            let max_scanner_distance = distances.max().unwrap();
            println!("{}", max_scanner_distance);
        }
    }
}
